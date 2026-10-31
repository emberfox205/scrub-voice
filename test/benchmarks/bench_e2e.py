"""End-to-End Pipeline Benchmark (Streaming Post-Speech Latency & Accuracy).

Evaluates the integrated edge voice system:
  Multi-channel capture -> Beamformer -> Denoiser -> ASR Decoder -> NLU Classifier -> IPC Dispatcher
Measures:
1. End-to-end post-speech latency percentiles (p50, p95, p99) against target budget (<= 500ms).
2. Per-subsystem latency breakdown (DSP, ASR finalize, NLU SetFit, Dispatcher IPC).
3. Ablation matrix: ours vs ablate_no_dsp vs ablate_bf_only vs ablate_dn_only.
4. Baseline comparisons (DeepFilterNet, noisereduce, open-vocab Vosk, Whisper).
5. Paired Wilcoxon signed-rank tests with Holm-Bonferroni correction.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
import sys
import time
from pathlib import Path
from time import perf_counter as pc
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from test.benchmarks.common.acoustics import (
    NoiseSource,
    ang_err,
    build_items,
    chunk_iter,
    circular_array,
    make_sample,
    quality,
    snr_label,
)
from test.benchmarks.common.registry import BUDGETS, Registry, RepoNLU
from test.benchmarks.common.reporting import _jd, make_figures, md_table, save_json_report, tex_table, write_csvs, write_tables
from test.benchmarks.common.rig import DispatcherRig
from test.benchmarks.common.stats import corpus_wer, holm, intent_metrics, norm_text, paired_test, slots_match, summarize, word_errors
from test.benchmarks.common.telemetry import ResourceSampler, hw_info


@dataclass
class Variant:
    name: str
    bf: str
    dn: str
    asr: str


def make_variants(args, reg: Registry) -> List[Variant]:
    ok = lambda k, n: reg.get(k, n) is not None
    bf = "repo_beamformer" if ok("bf", "repo_beamformer") else "ref_srp_das"
    dn = "repo_denoiser" if ok("dn", "repo_denoiser") else "spectral_gate"
    asr = "repo_asr" if ok("asr", "repo_asr") else "vosk_grammar"

    v = [
        Variant("ours", bf, dn, asr),
        Variant("ablate_no_dsp", "ref_channel", "passthrough", asr),
        Variant("ablate_bf_only", bf, "passthrough", asr),
        Variant("ablate_dn_only", "ref_channel", dn, asr),
    ]

    if getattr(args, "baselines", False):
        v += [
            Variant("base_deepfilternet", bf, "deepfilternet", asr),
            Variant("base_noisereduce", bf, "noisereduce", asr),
            Variant("base_vosk_openvocab", bf, dn, "vosk_open"),
            Variant("base_whisper", bf, dn, "whisper"),
        ]

    out = []
    for x in v:
        parts = [("bf", x.bf), ("dn", x.dn), ("asr", x.asr)]
        if all(ok(k, n) for k, n in parts):
            out.append(x)
        else:
            reg.skipped[f"variant:{x.name}"] = "; ".join(
                f"{k}:{n} -> {reg.skipped.get(f'{k}:{n}')}" for k, n in parts if not ok(k, n)
            )
    return out


def run_utterance(reg: Registry, var: Variant, nlu, rig, mc: np.ndarray, sr: int, chunk_s: float, hangover_ms: float) -> Dict[str, Any]:
    bf, dn, asr = reg.get("bf", var.bf), reg.get("dn", var.dn), reg.get("asr", var.asr)
    n = int(chunk_s * sr)
    asr.reset()
    t_bf, t_dn, t_asr, outs, azs = [], [], [], [], []

    for c in chunk_iter(mc, n):
        t0 = pc()
        y, az = bf.process(c, sr)
        t1 = pc()
        z = dn.process(y, sr)
        t2 = pc()
        asr.accept(z)
        t3 = pc()
        t_bf.append(t1 - t0)
        t_dn.append(t2 - t1)
        t_asr.append(t3 - t2)
        outs.append(z)
        if az is not None:
            azs.append(az)

    t0 = pc()
    text = asr.finalize()
    t_fin = pc() - t0

    t0 = pc()
    env = nlu.process(text)
    t_nlu = pc() - t0

    t_ipc = rig.roundtrip({**env, "timestamp": time.time()}) if (rig and env.get("status") == "EXECUTE") else None
    last_dsp = t_bf[-1] + t_dn[-1] + t_asr[-1]
    e2e = (last_dsp + t_fin + t_nlu) * 1000 + (t_ipc or 0.0) + hangover_ms

    return dict(
        text=text,
        env=env,
        e2e_ms=e2e,
        bf_ms=float(np.mean(t_bf) * 1000),
        dn_ms=float(np.mean(t_dn) * 1000),
        asr_accept_ms=float(np.mean(t_asr) * 1000),
        last_chunk_dsp_ms=float(last_dsp * 1000),
        asr_final_ms=float(t_fin * 1000),
        nlu_ms=float(t_nlu * 1000),
        ipc_ms=t_ipc,
        dsp_rtf=float((np.sum(t_bf) + np.sum(t_dn)) / (len(t_bf) * chunk_s)),
        out=np.concatenate(outs)[: mc.shape[1]],
        az=float(np.median(azs)) if azs else None,
    )


def bench_e2e(args, reg: Registry, nlu, rig, items, mic_xy: np.ndarray, noise: NoiseSource, snrs: List[float]) -> Dict[str, Any]:
    print("\n" + "=" * 72 + "\n [E2E] Streaming Pipeline Benchmark (Post-Speech Latency & Accuracy)\n" + "=" * 72)
    variants = make_variants(args, reg)
    if not variants or nlu is None:
        return {
            "status": "SKIPPED",
            "reason": "no runnable variant (wire --plugin / --vosk-model) or NLU missing",
            "skipped": reg.skipped,
        }

    labelled = [it for it in items if it.text] or items
    for k, v in reg.skipped.items():
        if k.startswith("variant:"):
            print(f"  [skip] {k}: {v}")

    # Warmup runs
    for x in variants:
        mc, _, _ = make_sample(labelled[0], 10.0, args, mic_xy, noise, np.random.default_rng(0))
        for _ in range(max(1, args.warmup // 2)):
            run_utterance(reg, x, nlu, rig, mc, args.sr, args.chunk_s, 0.0)

    recs: List[dict] = []
    for x in variants:
        for seed in range(args.seeds):
            rng = np.random.default_rng(3000 + seed)
            for snr in [None] + snrs:
                for it in labelled:
                    mc, clean, az = make_sample(it, snr, args, mic_xy, noise, rng)
                    r = run_utterance(reg, x, nlu, rig, mc, args.sr, args.chunk_s, args.vad_hangover_ms)
                    e, rw = word_errors(it.text, r["text"]) if it.text else (0, 0)
                    env, gt_ood = r["env"], it.status == "OUT-OF-DOMAIN"
                    ok = (
                        (env.get("status") == "OUT-OF-DOMAIN")
                        if gt_ood
                        else (env.get("intent") == it.intent and env.get("status") != "OUT-OF-DOMAIN")
                    )
                    rec = dict(
                        variant=x.name,
                        snr=snr_label(snr),
                        seed=seed,
                        item=it.name,
                        err=e,
                        ref=rw,
                        hyp=r["text"],
                        gt_status=it.status or "EXECUTE",
                        pred_status=env.get("status"),
                        intent_ok=bool(ok) if it.intent else None,
                        slot_ok=slots_match(it.slots, env.get("slots") or {}) if it.slots else None,
                        **{
                            k: r[k]
                            for k in (
                                "e2e_ms",
                                "bf_ms",
                                "dn_ms",
                                "asr_accept_ms",
                                "last_chunk_dsp_ms",
                                "asr_final_ms",
                                "nlu_ms",
                                "ipc_ms",
                                "dsp_rtf",
                            )
                        },
                    )
                    if clean is not None:
                        rec["si_sdr_out"] = quality(clean, r["out"], args.sr, False)["si_sdr"]
                    if r["az"] is not None and az is not None:
                        rec["doa_err"] = ang_err(r["az"], az)
                    recs.append(rec)

    out: Dict[str, Any] = {
        "variants": [v.__dict__ for v in variants],
        "skipped": reg.skipped,
        "summary": {},
        "paired": {},
    }
    keys = sorted({(r["variant"], r["snr"]) for r in recs})

    for var, snr in keys:
        rs = [r for r in recs if r["variant"] == var and r["snr"] == snr]
        m: Dict[str, Any] = {
            "e2e_ms": summarize([r["e2e_ms"] for r in rs]),
            "stage_mean_ms": {
                k: float(np.mean([r[k] for r in rs]))
                for k in ("bf_ms", "dn_ms", "asr_accept_ms", "asr_final_ms", "nlu_ms")
            },
            "dsp_rtf": summarize([r["dsp_rtf"] for r in rs]),
        }
        ipc = [r["ipc_ms"] for r in rs if r["ipc_ms"] is not None]
        m["ipc_ms"] = summarize(ipc) if ipc else None

        if any(r["ref"] for r in rs):
            m["wer"] = corpus_wer([r["err"] for r in rs], [r["ref"] for r in rs])
        irs = [dict(r, gt_status=r["gt_status"]) for r in rs if r["intent_ok"] is not None]
        if irs:
            m["nlu"] = intent_metrics(irs)
        if any("si_sdr_out" in r for r in rs):
            m["si_sdr_out"] = summarize([r.get("si_sdr_out") for r in rs if r.get("si_sdr_out") is not None])
        if any("doa_err" in r for r in rs):
            m["doa_err_deg"] = summarize([r.get("doa_err") for r in rs if r.get("doa_err") is not None])

        m["status"] = "PASS" if m["e2e_ms"]["p95"] <= BUDGETS["e2e_ms_max"] else "FAIL"
        out["summary"][f"{var}/{snr}"] = m

        w = f"WER {100 * m['wer']['wer']:5.1f}%" if "wer" in m else "WER   n/a"
        a = f"intent {100 * m['nlu']['intent_acc']['p']:5.1f}%" if "nlu" in m else ""
        print(
            f"  {var:<22} {snr:<6} e2e p50 {m['e2e_ms']['p50']:7.1f}  p95 {m['e2e_ms']['p95']:7.1f}  "
            f"p99 {m['e2e_ms']['p99']:7.1f} ms  {w}  {a}  [{m['status']}]"
        )

    # Paired comparisons vs "ours", Holm-corrected within each metric
    for metric, field_ in (("e2e_ms", "e2e_ms"), ("word_errors", "err")):
        res = {}
        for var, snr in keys:
            if var == "ours":
                continue
            a_dict = {(r["seed"], r["item"]): r[field_] for r in recs if r["variant"] == "ours" and r["snr"] == snr}
            b_dict = {(r["seed"], r["item"]): r[field_] for r in recs if r["variant"] == var and r["snr"] == snr}
            common = sorted(set(a_dict) & set(b_dict))
            if common:
                res[f"{var}/{snr}"] = paired_test([a_dict[k] for k in common], [b_dict[k] for k in common])
        ps = [v["p"] for v in res.values() if v.get("p") is not None]
        if ps:
            adj = iter(holm(ps))
            for v in res.values():
                v["p_holm"] = next(adj) if v.get("p") is not None else None
        out["paired"][metric] = res

    out["raw_rows"] = recs
    return out


def parse_args():
    p = argparse.ArgumentParser(description="SCRUB-VOICE End-to-End Benchmark")
    p.add_argument("--manifest", help="JSONL labelled dataset with audio paths")
    p.add_argument("--wav", help="Single WAV file")
    p.add_argument("--synthetic", type=int, default=2, help="N synthetic speech clips")
    p.add_argument("--noise-wav", help="Noise WAV for SNR sweeps")
    p.add_argument("--snr-sweep", default="15,0")
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-items", type=int, default=0)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--chunk-s", type=float, default=0.5)
    p.add_argument("--mic-channels", default="0,1,2,3,4,5")
    p.add_argument("--mic-radius", type=float, default=0.043)
    p.add_argument("--diffuse-ratio", type=float, default=0.5)
    p.add_argument("--beamformers", default="ref_channel,ref_srp_das,repo_beamformer")
    p.add_argument("--denoisers", default="passthrough,spectral_gate,repo_denoiser")
    p.add_argument("--asrs", default="vosk_grammar,repo_asr")
    p.add_argument("--baselines", action="store_true")
    p.add_argument("--plugin", action="append", default=[])
    default_vosk = REPO_ROOT / "src" / "asr" / "models" / "vosk-model-small-en-us-0.15"
    p.add_argument("--vosk-model", default=str(default_vosk) if default_vosk.exists() else None)
    p.add_argument("--grammar", help="Grammar JSON or phrase file")
    p.add_argument("--whisper-model", default="tiny.en")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--vad-hangover-ms", type=float, default=0.0)
    p.add_argument("--no-dispatcher", action="store_true")
    p.add_argument("--out", default="", help="Output directory")
    p.add_argument("--tag", default="")
    a = p.parse_args()
    a.mic_idx = [int(x) for x in a.mic_channels.split(",")]
    a.beamformers = [s for s in a.beamformers.split(",") if s]
    a.denoisers = [s for s in a.denoisers.split(",") if s]
    a.asrs = [s for s in a.asrs.split(",") if s]
    return a


def main():
    args = parse_args()
    print("*" * 72 + "\n SCRUB-VOICE BENCHMARK | System-Level: End-to-End Pipeline\n" + "*" * 72)
    rng = np.random.default_rng(args.seed)
    snrs = [float(s) for s in args.snr_sweep.split(",") if s]
    mic_xy = circular_array(len(args.mic_idx), args.mic_radius)
    noise = NoiseSource(args.noise_wav, args.sr)
    items = build_items(args, rng)

    nlu = None
    try:
        nlu = RepoNLU(args.threads)
    except Exception as e:
        print(f"  [warn] Could not load RepoNLU ({e})")

    grammar = None
    if getattr(args, "grammar", None):
        t = Path(args.grammar).read_text()
        try:
            grammar = json.loads(t)
        except Exception:
            grammar = [ln.strip() for ln in t.splitlines() if ln.strip()]

    reg = Registry(args, mic_xy, grammar)

    rig = None
    if nlu is not None and not args.no_dispatcher:
        try:
            rig = DispatcherRig("e2e")
        except Exception as e:
            print(f"  [warn] Dispatcher rig unavailable ({e})")

    with ResourceSampler() as rs:
        res = bench_e2e(args, reg, nlu, rig, items, mic_xy, noise, snrs)
    res["resources"] = rs.summary()

    if rig:
        rig.close()

    if args.out:
        outdir = Path(args.out) / (f"bench_e2e_{time.strftime('%Y%m%d_%H%M%S')}" + (f"_{args.tag}" if args.tag else ""))
        outdir.mkdir(parents=True, exist_ok=True)
        report = {
            "timestamp": time.time(),
            "hardware": hw_info(),
            "stages": {"e2e": res},
        }
        save_json_report(report, outdir / "report_e2e.json")
        write_csvs(report, outdir)
        write_tables(report, outdir)
        make_figures(report, outdir)
        print(f"\n Results saved to: {outdir}")


if __name__ == "__main__":
    main()

