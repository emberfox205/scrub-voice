"""Isolated NLU Subsystem Benchmark (Intent Classification & Slot Extraction).

Evaluates:
1. Intent classification accuracy, 95% Wilson confidence intervals.
2. Out-of-Domain (OOD) False Accept Rate (FAR) and In-Domain False Reject Rate (FRR).
3. Exact slot matching rate.
4. Schema validation conformance against Draft 2020-12 command_schema.json.
5. Inference latency percentiles (p50, p95, p99) against target budget (<= 80ms).
6. Optional TF-IDF + LogisticRegression baseline comparison with paired Wilcoxon signed-rank test.
7. Optional PyTorch CPU thread scaling sweep (1 thread pinned, 2 threads, 3 threads).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from time import perf_counter as pc
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from test.benchmarks.common.registry import BUDGETS, DEFAULT_NLU_EVAL, RepoNLU, TfidfNLU, _try_import
from test.benchmarks.common.reporting import _jd, md_table, save_json_report, write_csvs
from test.benchmarks.common.stats import paired_test, slots_match, summarize, wilson
from test.benchmarks.common.telemetry import ResourceSampler, get_rss_mb, hw_info


def _intent_metrics(rows: List[dict]) -> Dict[str, Any]:
    n = len(rows)
    ood = [r for r in rows if r["gt_status"] == "OUT-OF-DOMAIN"]
    ind = [r for r in rows if r["gt_status"] != "OUT-OF-DOMAIN"]
    acc = wilson(sum(r["intent_ok"] for r in rows), n)
    far = wilson(sum(r["pred_status"] == "EXECUTE" for r in ood), len(ood))
    frr = wilson(sum(r["pred_status"] == "OUT-OF-DOMAIN" for r in ind), len(ind))
    slot_rows = [r for r in ind if r["slot_ok"] is not None]
    slot = wilson(sum(r["slot_ok"] for r in slot_rows), len(slot_rows))
    return {
        "n": n,
        "intent_acc": {"p": acc[0], "lo": acc[1], "hi": acc[2]},
        "ood_false_accept": {"p": far[0], "lo": far[1], "hi": far[2], "n": len(ood)},
        "in_domain_false_reject": {"p": frr[0], "lo": frr[1], "hi": frr[2], "n": len(ind)},
        "slot_exact": {"p": slot[0], "lo": slot[1], "hi": slot[2], "n": len(slot_rows)},
    }


def _eval_nlu(nlu, data: List[dict], reps: int, warmup: int, schema=None) -> Tuple[List[dict], List[float]]:
    jsonschema = _try_import("jsonschema")
    for d in data[: max(1, warmup)]:
        nlu.process(d["text"])
    rows, lat = [], []
    for d in data:
        env = None
        for _ in range(reps):
            t0 = pc()
            env = nlu.process(d["text"])
            lat.append((pc() - t0) * 1000)
        valid = None
        if schema is not None and jsonschema:
            try:
                jsonschema.validate(instance=env, schema=schema)
                valid = True
            except Exception:
                valid = False
        gt_ood = d["status"] == "OUT-OF-DOMAIN"
        pred_intent = env.get("intent")
        pred_status = env.get("status")

        if gt_ood:
            intent_ok = pred_status == "OUT-OF-DOMAIN"
        else:
            intent_ok = (pred_intent == d["intent"]) and (pred_status != "OUT-OF-DOMAIN")

        rows.append(
            dict(
                text=d["text"],
                gt_intent=d["intent"],
                gt_status=d["status"],
                pred_intent=pred_intent,
                pred_status=pred_status,
                intent_ok=intent_ok,
                slot_ok=slots_match(d.get("slots") or {}, env.get("slots") or {}) if d.get("slots") else None,
                schema_valid=valid,
            )
        )
    return rows, lat


def load_nlu_eval(args, items=None) -> List[dict]:
    if getattr(args, "nlu_eval", None):
        return [json.loads(line) for line in Path(args.nlu_eval).read_text().splitlines() if line.strip()]
    if items:
        labelled = [
            dict(text=i.text, intent=i.intent, status=i.status or "EXECUTE", slots=i.slots)
            for i in items
            if getattr(i, "text", None) and getattr(i, "intent", None)
        ]
        if labelled:
            return labelled
    return [dict(text=t, intent=i, status=s, slots={}) for t, i, s in DEFAULT_NLU_EVAL]


def bench_nlu(args, items=None) -> Dict[str, Any]:
    print("\n" + "=" * 72 + "\n [NLU] Intent Classification, Schema Validation & Thread Scaling\n" + "=" * 72)
    data = load_nlu_eval(args, items)
    if len(data) < 100:
        print(f"  [!] only {len(data)} NLU test utterances: CIs will be wide; use >=300 incl. >=100 OOD for publication.")
    out: Dict[str, Any] = {"n_eval": len(data)}
    try:
        r0 = get_rss_mb()
        t0 = pc()
        nlu = RepoNLU(args.threads)
        load_s, rss_d = pc() - t0, get_rss_mb() - r0
    except Exception as e:
        return {"status": "SKIPPED", "reason": f"{type(e).__name__}: {e}"}

    rows, lat = _eval_nlu(nlu, data, args.reps_nlu, args.warmup, nlu.schema)
    m = _intent_metrics(rows)
    lat_s = summarize(lat)
    valid = [r["schema_valid"] for r in rows if r["schema_valid"] is not None]
    out["repo_setfit"] = {
        **m,
        "latency_ms": lat_s,
        "load_s": round(load_s, 2),
        "rss_delta_mb": round(rss_d, 1),
        "schema_valid_rate": float(np.mean(valid)) if valid else None,
        "status": "PASS" if lat_s["p95"] <= BUDGETS["nlu_ms_max"] else "FAIL",
    }
    out["errors"] = [r for r in rows if not r["intent_ok"]][:25]
    print(
        f"  repo_setfit  acc {100 * m['intent_acc']['p']:.1f}% [{100 * m['intent_acc']['lo']:.1f},{100 * m['intent_acc']['hi']:.1f}]  "
        f"OOD-FAR {100 * m['ood_false_accept']['p']:.1f}% (n={m['ood_false_accept']['n']})  lat p95 {lat_s['p95']:.1f} ms  "
        f"[{out['repo_setfit']['status']}]"
    )

    if getattr(args, "nlu_train", None) and _try_import("sklearn"):
        train = [json.loads(line) for line in Path(args.nlu_train).read_text().splitlines() if line.strip()]
        b = TfidfNLU(train)
        brows, blat = _eval_nlu(b, data, args.reps_nlu, args.warmup)
        bm = _intent_metrics(brows)
        out["tfidf_logreg"] = {**bm, "latency_ms": summarize(blat)}
        print(
            f"  tfidf_logreg acc {100 * bm['intent_acc']['p']:.1f}%  OOD-FAR {100 * bm['ood_false_accept']['p']:.1f}%  "
            f"lat p95 {out['tfidf_logreg']['latency_ms']['p95']:.2f} ms"
        )
        pt = paired_test([float(r["intent_ok"]) for r in rows], [float(r["intent_ok"]) for r in brows])
        out["setfit_vs_tfidf_paired"] = pt

    if getattr(args, "nlu_thread_sweep", False):
        out["thread_sweep"] = {}
        for th, pin in [(1, True), (2, False), (3, False)]:
            old = None
            if pin and hasattr(os, "sched_setaffinity"):
                old = os.sched_getaffinity(0)
                os.sched_setaffinity(0, {sorted(old)[0]})
            try:
                p = RepoNLU(th)
                _, l = _eval_nlu(p, data, args.reps_nlu, args.warmup)
                out["thread_sweep"][f"{th}thr_{'pinned' if pin else 'free'}"] = summarize(l)
                print(f"  threads={th} pinned={pin}: p50 {np.percentile(l, 50):.1f} ms  p95 {np.percentile(l, 95):.1f} ms")
            except Exception as e:
                out["thread_sweep"][f"{th}thr"] = {"error": str(e)}
            finally:
                if old:
                    os.sched_setaffinity(0, old)

    out["raw_rows"] = rows
    return out


def parse_args():
    p = argparse.ArgumentParser(description="SCRUB-VOICE isolated NLU benchmark")
    p.add_argument("--nlu-eval", help="JSONL evaluation dataset: {text, intent, status, slots}")
    p.add_argument("--nlu-train", help="JSONL training set for TF-IDF baseline")
    p.add_argument("--nlu-thread-sweep", action="store_true", help="Sweep 1, 2, 3 PyTorch threads")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--reps-nlu", type=int, default=5)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--out", default="", help="Output directory for reports")
    p.add_argument("--tag", default="")
    return p.parse_args()


def main():
    args = parse_args()
    print("*" * 72 + "\n SCRUB-VOICE BENCHMARK | Subsystem: NLU (SetFit & Schema Validation)\n" + "*" * 72)
    with ResourceSampler() as rs:
        res = bench_nlu(args)
    res["resources"] = rs.summary()

    if args.out:
        outdir = Path(args.out) / (f"bench_nlu_{time.strftime('%Y%m%d_%H%M%S')}" + (f"_{args.tag}" if args.tag else ""))
        outdir.mkdir(parents=True, exist_ok=True)
        report = {
            "timestamp": time.time(),
            "hardware": hw_info(),
            "stages": {"nlu": res},
        }
        save_json_report(report, outdir / "report_nlu.json")
        write_csvs(report, outdir)
        print(f"\n Results saved to: {outdir}")


if __name__ == "__main__":
    main()

