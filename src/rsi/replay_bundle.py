"""SHA-256 signed ReplayBundle for the Bi-Level Harness RSI runs.

Layout (outputs/replay_bundle/):
  <program>/<arm>/evolution_trace.json   per-generation events, GUARD decisions, candidate harness genomes, chain hashes
  <program>/<arm>/sim_memory.json        SIM lessons written by the outer loop
  <program>/<arm>/final_harness.json     promoted harness H* (elective pool, capstone packages, syllabus modules, policy)
  <program>/<arm>/qwen_calls.jsonl       exact ledger entries (prompt preview + response) for every Qwen call of the run
  harnesspro_pareto_archive.json         Lite / Medium / Heavy archive of the primary case
  MANIFEST.json                          SHA-256 of every file, the master_sha256 it belongs to, and a bundle root hash
The bundle root hash is SHA-256 over the sorted "path:sha256" lines, so any edit to any file is detectable.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict


def _dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


def export_replay_bundle(out_dir: Path, program_rsi_runs: Dict[str, Dict[str, Any]], pareto_archive: Dict[str, Any],
                         master_sha256: str, ledger_path: Path) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    ledger = {}
    if Path(ledger_path).exists():
        for line in Path(ledger_path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                ledger.setdefault(rec.get("request_hash"), rec)
    for pid, runs in program_rsi_runs.items():
        for arm, run in runs.items():
            d = out_dir / pid / arm
            _dump(d / "evolution_trace.json", {"program_id": pid, "mode": run["mode"], "generations_run": run["generations_run"],
                                               "accepted_promotions": run["accepted_promotions"], "guard_rollbacks": run["guard_rollbacks"],
                                               "trace_final_sha256": run["trace_final_sha256"], "history": run["evolution_history"]})
            _dump(d / "sim_memory.json", run["sim_memory"])
            tb = run["transfer_bundle"]
            _dump(d / "final_harness.json", {"elective_pool": tb["elective_pool"], "capstone_packages": tb["capstone_packages"],
                                             "syllabus_modules": tb["syllabus_modules"], "policy": tb["full_policy"]})
            lines = []
            for a in run["qwen_audits"]:
                rec = ledger.get(a.get("request_hash"))
                lines.append(json.dumps(rec if rec else {"request_hash": a.get("request_hash"), "missing_in_ledger": True}, ensure_ascii=False))
            (d / "qwen_calls.jsonl").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    _dump(out_dir / "harnesspro_pareto_archive.json", pareto_archive)
    files = sorted(p for p in out_dir.rglob("*") if p.is_file())
    entries = {p.relative_to(out_dir).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    root = hashlib.sha256("\n".join(f"{k}:{v}" for k, v in sorted(entries.items())).encode("utf-8")).hexdigest()
    manifest = {"master_sha256": master_sha256, "files": entries, "bundle_root_sha256": root, "n_files": len(entries),
                "root_rule": "sha256 over sorted 'relative_path:sha256' lines joined by \\n"}
    _dump(out_dir / "MANIFEST.json", manifest)
    return manifest


def verify_replay_bundle(out_dir: Path) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    man = json.loads((out_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    bad = [k for k, v in man["files"].items() if hashlib.sha256((out_dir / k).read_bytes()).hexdigest() != v]
    root = hashlib.sha256("\n".join(f"{k}:{v}" for k, v in sorted(man["files"].items())).encode("utf-8")).hexdigest()
    return {"files_ok": not bad, "tampered": bad, "root_ok": root == man["bundle_root_sha256"], "master_sha256": man["master_sha256"]}
