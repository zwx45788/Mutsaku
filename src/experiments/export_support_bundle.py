"""Export the compact, de-identified verification inputs used by verify_key_results.py.

Outputs (under evoalign_os/data/verification/):
  features_{split}.npz      per-JD 48-skill hit matrix, employer-balanced weights, hashed employer ids,
                            JD Bloom targets and track labels for dev_2024 / replay_2025 / replay_2026q1
  task1_targets.npz         2024 dev demand distributions for the three role populations used in Task 1
  benchmark_pathways.json   reconciled expected-credit pathways of the 11 benchmark programmes
  kg_min.json               48-skill ids/names/stages + semantic ground-distance matrix
  mpc_yearly_counts.csv     2018-2024 yearly bachelor-accessible ad and employer counts
  sample_jd_input.csv / sample_jd_output.csv   30 de-identified JD rows -> extracted skills + track
No raw JD text beyond the 30-row sample is exported; employer keys are re-hashed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import duckdb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analytics.gap_engine import _compile_skill_regexes, compute_supply_profile, extract_demand_profile, load_curricula, load_kg
from analytics.pathways import expected_pathway, group_curricula, tier_default
from analytics.track_evaluator import TRACK_IDS, build_track_demand, classify_jd_track
from security.leakage_guard import BlindAuditContext

ROOT = Path("evoalign_os")
PROC = ROOT / "data" / "processed"
BLIND = ROOT / "data" / "blind_holdout"
VER = ROOT / "data" / "verification"
BENCH = ("PKU_CS_2024", "U03_BUAA_CS_2025", "BJTU_CS_2025", "BJTU_AI_2025", "CUP_CS_2024", "BTBU_CS_2022",
         "BTBU_AI_2022", "BTBU_SE_2022", "BTBU_AUTO_2024", "BJUT_CS_SYLLABI", "BISTU_CS_2025")
KEEP = ("course_name", "credits", "take_rate", "category", "required", "semester", "lab_hours", "is_core_foundation", "extracted_skills")


def _h(s: str) -> str:
    return hashlib.sha256(("evoalign-verify:" + s).encode()).hexdigest()[:16]


def _save_split(name: str, td: dict) -> None:
    lab = {tk: i for i, tk in enumerate(TRACK_IDS)}
    np.savez_compressed(
        VER / f"features_{name}.npz",
        hits=td["hits"].astype(np.uint8),
        ad_weights=td["ad_weights"],
        # Order-preserving pseudonyms (rank of the original key) keep bootstrap cluster order identical
        employers=np.array([f"E{i:06d}" for i in np.unique(np.asarray(td["employers"]).astype(str), return_inverse=True)[1]]),
        job_bloom=td["job_bloom"],
        demand_dist=td["demand_dist"],
        track_labels=np.array([lab[t] for t in td["track_labels"]], dtype=np.int8),
        track_weight_share=np.array([td["tracks"][tk]["weight_share"] for tk in TRACK_IDS]),
        track_demand_dist=np.stack([td["tracks"][tk]["demand_dist"] for tk in TRACK_IDS]),
        track_n_ads=np.array([td["tracks"][tk]["n_ads"] for tk in TRACK_IDS]),
        n_employers=np.array(td["n_employers"]),
    )


def export_features() -> None:
    VER.mkdir(parents=True, exist_ok=True)
    kg = load_kg()
    dev = extract_demand_profile([PROC / "demand_dev_2024.parquet"], kg, use_bj_shortage_calibration=False)
    _save_split("dev_2024", build_track_demand(dev, kg))
    with BlindAuditContext("export de-identified verification features for support package", caller="export_support_bundle"):
        r25 = extract_demand_profile([BLIND / "demand_holdout_2025.parquet"], kg, population="broad_entry", use_bj_shortage_calibration=False)
        r26 = extract_demand_profile([BLIND / "demand_drift_2026q1.parquet"], kg, population="broad_entry", use_bj_shortage_calibration=False)
    _save_split("replay_2025", build_track_demand(r25, kg))
    _save_split("replay_2026q1", build_track_demand(r26, kg))

    # Task 1 targets (identical calls to run_task1_benchmark: default calibration)
    tgt = {"broad_entry": extract_demand_profile([PROC / "demand_dev_2024.parquet"], kg, population="broad_entry")}
    for pop in ("research", "technical"):
        tgt[pop] = extract_demand_profile([PROC / "demand_dev_2024.parquet"], kg, population=pop)
    np.savez_compressed(VER / "task1_targets.npz", **{p: d["demand_dist"] for p, d in tgt.items()},
                        **{f"{p}_n": np.array(d["n_rows"]) for p, d in tgt.items()})

    groups = group_curricula(load_curricula())
    paths = {}
    for pid in BENCH:
        pw, audit = expected_pathway(groups[pid])
        paths[pid] = {
            "tier": groups[pid][0]["tier"], "role_target": tier_default(groups[pid][0]["tier"]),
            "pathway_audit": audit,
            "courses": [{k: c.get(k) for k in KEEP if k in c} for c in pw],
        }
    (VER / "benchmark_pathways.json").write_text(json.dumps(paths, ensure_ascii=False), encoding="utf-8")
    (VER / "kg_min.json").write_text(json.dumps({
        "skill_ids": kg["skill_ids"],
        "skills": [{k: s[k] for k in ("id", "zh", "domain", "stage", "job_bloom")} for s in kg["skills"]],
        "semantic_ground_distance": {"matrix": kg["semantic_ground_distance"]["matrix"]},
    }, ensure_ascii=False), encoding="utf-8")

    con = duckdb.connect()
    rows = con.execute("""
        SELECT year, COUNT(*) AS n_ads, COUNT(DISTINCT employer_key) AS n_emp FROM (
          SELECT year, employer_key FROM read_parquet('evoalign_os/data/processed/demand_train_2018_2023.parquet') WHERE broad_bachelor_accessible_entry = TRUE
          UNION ALL
          SELECT year, employer_key FROM read_parquet('evoalign_os/data/processed/demand_dev_2024.parquet') WHERE broad_bachelor_accessible_entry = TRUE)
        GROUP BY year ORDER BY year""").fetchall()
    sample = con.execute("""
        SELECT * FROM (
          SELECT DISTINCT ON (job_key) job_key, job_title, education_band, experience_raw, salary_min, salary_max, description_redacted
          FROM read_parquet('evoalign_os/data/processed/demand_dev_2024.parquet')
          WHERE broad_bachelor_accessible_entry = TRUE AND description_characters BETWEEN 150 AND 700)
        ORDER BY hash(job_key) LIMIT 30""").fetchall()
    con.close()
    with (VER / "mpc_yearly_counts.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["year", "n_ads", "n_employers"])
        w.writerows(rows)

    regexes = _compile_skill_regexes(kg["skills"], None)
    zh = {s["id"]: s["zh"] for s in kg["skills"]}
    with (VER / "sample_jd_input.csv").open("w", newline="", encoding="utf-8-sig") as fi, \
         (VER / "sample_jd_output.csv").open("w", newline="", encoding="utf-8-sig") as fo:
        wi, wo = csv.writer(fi), csv.writer(fo)
        wi.writerow(["job_id", "job_title", "education", "experience", "salary_min", "salary_max", "description_redacted"])
        wo.writerow(["job_id", "track", "n_skills", "skill_ids", "skill_names"])
        for jk, title, edu, exp, smin, smax, desc in sample:
            jid = _h(jk)
            text = f"{title or ''} {desc or ''}"
            hit = np.array([1 if pat.search(text) else 0 for _, pat in regexes], dtype=np.uint8)
            sids = [kg["skill_ids"][j] for j in np.nonzero(hit)[0]]
            wi.writerow([jid, title, edu, exp, smin, smax, (desc or "").replace("\n", " ")])
            wo.writerow([jid, classify_jd_track(title, hit, kg["skill_ids"]), len(sids), ";".join(sids), ";".join(zh[s] for s in sids)])
    (VER / "employment_feedback_series.json").write_bytes((PROC / "employment_feedback_series.json").read_bytes())
    print("verification features exported to", VER)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-only", action="store_true")
    ap.parse_args()
    export_features()
