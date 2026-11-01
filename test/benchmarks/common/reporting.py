"""Reporting, export formatters, LaTeX table generation, and visualization.

Generates:
- report.json: Complete telemetry and statistical metrics.
- raw_<stage>.csv: Measurement-level CSVs for analysis in R/pandas.
- tables.md and tables.tex: Academic booktabs tables for papers.
- CDF plots: Post-speech latency cumulative distribution and WER vs SNR curves.
"""
from __future__ import annotations

import csv
import importlib
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from test.benchmarks.common.acoustics import safe_ram_first_save
from test.benchmarks.common.registry import BUDGETS

def _try_import(name: str):
    try:
        return importlib.import_module(name)
    except Exception:
        return None


def _jd(o: Any) -> Any:
    """JSON serialization helper for numpy types."""
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def md_table(h: List[str], rows: List[List[str]]) -> str:
    """Formats markdown table."""
    return "\n".join(["| " + " | ".join(h) + " |", "|" + "|".join("---" for _ in h) + "|"] + ["| " + " | ".join(r) + " |" for r in rows])


def tex_table(h: List[str], rows: List[List[str]], caption: str, label: str) -> str:
    """Formats academic LaTeX table using booktabs."""
    esc = lambda s: s.replace("_", r"\_").replace("%", r"\%")
    body = "\n".join(" & ".join(esc(c) for c in r) + r" \\" for r in rows)
    return (
        f"\\begin{{table}}[t]\n\\centering\\small\n\\caption{{{caption}}}\\label{{{label}}}\n"
        f"\\begin{{tabular}}{{l{'r' * (len(h) - 1)}}}\n\\toprule\n"
        + " & ".join(esc(x) for x in h)
        + " \\\\\n\\midrule\n"
        + body
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


def save_json_report(report: Dict[str, Any], target_path: Path) -> Path:
    """Saves benchmark report JSON to RAM first, then safely persists to target_path."""
    def _dump(dest: Path):
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=_jd)
    return safe_ram_first_save(Path(target_path), _dump)


def write_tables(report: Dict[str, Any], outdir: Path):
    """Writes markdown and LaTeX tables to outdir (RAM-buffered)."""
    md, tex = [], []
    e2e = report["stages"].get("e2e", {})
    if e2e.get("summary"):
        h = ["Variant", "SNR", "E2E p50 (ms)", "p95", "p99", "WER (%)", "Intent acc (%)", "OOD FAR (%)"]
        rows = []
        for k, m in e2e["summary"].items():
            v, s = k.split("/")
            rows.append([
                v, s,
                f"{m['e2e_ms']['p50']:.0f}",
                f"{m['e2e_ms']['p95']:.0f}",
                f"{m['e2e_ms']['p99']:.0f}",
                f"{100 * m['wer']['wer']:.1f}" if "wer" in m else "-",
                f"{100 * m['nlu']['intent_acc']['p']:.1f}" if "nlu" in m else "-",
                f"{100 * m['nlu']['ood_false_accept']['p']:.1f}" if "nlu" in m and m["nlu"]["ood_false_accept"]["n"] else "-",
            ])
        md += ["### End-to-end, post-speech", md_table(h, rows)]
        tex.append(tex_table(h, rows, "End-to-end post-speech latency and accuracy on Raspberry Pi 4.", "tab:e2e"))
    dsp = report["stages"].get("dsp", {}).get("timing", {})
    if dsp:
        h = ["Component", "RTF mean", "RTF p95", "RTF p99", "Deadline miss (%)", "Status"]
        rows = [
            [k, f"{v['rtf']['mean']:.3f}", f"{v['rtf']['p95']:.3f}", f"{v['rtf']['p99']:.3f}", f"{100 * v['deadline_miss_rate']:.1f}", v["status"]]
            for k, v in dsp.items()
        ]
        md += ["### DSP real-time factor", md_table(h, rows)]
        tex.append(tex_table(h, rows, "DSP real-time factor per 0.5 s chunk (single process, Raspberry Pi 4).", "tab:dsp"))
    
    md_content = "\n\n".join(md) + "\n"
    tex_content = "\n".join(tex)
    safe_ram_first_save(outdir / "tables.md", lambda p: p.write_text(md_content, encoding="utf-8"))
    safe_ram_first_save(outdir / "tables.tex", lambda p: p.write_text(tex_content, encoding="utf-8"))


def write_csvs(report: Dict[str, Any], outdir: Path):
    """Exports raw measurement-level rows to CSV (RAM-buffered)."""
    for stage, st in report["stages"].items():
        rows = st.get("raw_rows") if isinstance(st, dict) else None
        if rows:
            keys = sorted({k for r in rows for k in r})
            def _write_csv(dest: Path, _keys=keys, _rows=rows):
                with open(dest, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=_keys)
                    w.writeheader()
                    w.writerows(_rows)
            safe_ram_first_save(outdir / f"raw_{stage}.csv", _write_csv)


def make_figures(report: Dict[str, Any], outdir: Path):
    """Generates CDF latency plots and WER vs SNR curves (RAM-buffered)."""
    plt = _try_import("matplotlib.pyplot")
    if plt is None:
        return
    e2e = report["stages"].get("e2e", {})
    rows = e2e.get("raw_rows") or []
    if rows:
        fig, ax = plt.subplots(figsize=(5, 3.2))
        for v in sorted({r["variant"] for r in rows}):
            x = np.sort([r["e2e_ms"] for r in rows if r["variant"] == v])
            ax.plot(x, np.arange(1, len(x) + 1) / len(x), label=v)
        ax.axvline(BUDGETS["e2e_ms_max"], ls="--", c="k", lw=0.8)
        ax.set(xlabel="post-speech latency (ms)", ylabel="CDF")
        ax.legend(fontsize=6)
        fig.tight_layout()
        safe_ram_first_save(outdir / "fig_e2e_latency_cdf.pdf", lambda p: fig.savefig(p))
        safe_ram_first_save(outdir / "fig_e2e_latency_cdf.png", lambda p: fig.savefig(p, dpi=200))
        plt.close(fig)

        snr_order = [s for s in ["clean", "20dB", "15dB", "10dB", "5dB", "0dB"] if any(r["snr"] == s for r in rows)]
        if len(snr_order) > 1 and any(r["ref"] for r in rows):
            fig, ax = plt.subplots(figsize=(5, 3.2))
            for v in sorted({r["variant"] for r in rows}):
                ys = []
                for s in snr_order:
                    rs = [r for r in rows if r["variant"] == v and r["snr"] == s]
                    ys.append(100 * sum(r["err"] for r in rs) / max(1, sum(r["ref"] for r in rs)))
                ax.plot(snr_order, ys, marker="o", label=v)
            ax.set(xlabel="SNR", ylabel="WER (%)")
            ax.legend(fontsize=6)
            fig.tight_layout()
            safe_ram_first_save(outdir / "fig_wer_vs_snr.pdf", lambda p: fig.savefig(p))
            safe_ram_first_save(outdir / "fig_wer_vs_snr.png", lambda p: fig.savefig(p, dpi=200))
            plt.close(fig)


def print_budget_table(report: Dict[str, Any]):
    """Prints a terminal summary table of PASS/FAIL status against architectural budgets."""
    print("\n" + "=" * 72 + "\n BUDGET SUMMARY\n" + "=" * 72)
    S = report["stages"]
    checks = []
    for k, v in S.get("dsp", {}).get("timing", {}).items():
        checks.append((f"DSP {k} RTF p95", f"{v['rtf']['p95']:.3f}", f"<= {BUDGETS['dsp_rtf_max']}", v["status"]))
    n = S.get("nlu", {}).get("repo_setfit")
    if n:
        checks.append(("NLU latency p95 (ms)", f"{n['latency_ms']['p95']:.1f}", f"<= {BUDGETS['nlu_ms_max']}", n["status"]))
    i = S.get("ipc", {})
    if i.get("transport_status"):
        checks.append(("IPC transport", "-", f"p95<={BUDGETS['ipc_ms_max']} ms, >5k/s", i["transport_status"]))
    d = i.get("dispatcher", {})
    for key, label in (("roundtrip_status", "Dispatcher round trip"), ("preemption_status", "Emergency preemption"), ("malformed_status", "Malformed payloads")):
        if key in d:
            checks.append((label, "-", "-", d[key]))
    e = S.get("e2e", {}).get("summary", {})
    if "ours/clean" in e:
        checks.append(("E2E p95 clean (ms)", f"{e['ours/clean']['e2e_ms']['p95']:.0f}", f"<= {BUDGETS['e2e_ms_max']}", e["ours/clean"]["status"]))
    peak = report.get("resources", {}).get("peak_rss_mb")
    if peak:
        checks.append(("Total peak RSS (MB)", f"{peak:.0f}", f"<= {BUDGETS['rss_total_mb_max']:.0f}", "PASS" if peak <= BUDGETS["rss_total_mb_max"] else "FAIL"))
    for lab, val, bud, st in checks:
        print(f"  {lab:<30} {val:>10}   budget {bud:<22} {st}")

