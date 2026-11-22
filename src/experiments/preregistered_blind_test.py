"""Preregistered, one-shot independent blind test (addresses the fact that the Beijing 2025 replay was seen during development).

Step 1  `register`: freezes the evaluation — records SHA-256 of the sealed master results, the per-track plan snapshots,
        the knowledge graph and the evaluator source files, the exact data-selection SQL, the hypotheses and the decision
        rules — BEFORE any row of the blind populations is extracted. Writes outputs/preregistration_blind_v3.json.
Step 2  `run`: verifies every frozen hash still matches, extracts the never-used populations from the official recruitment
        catalogue into data/blind_holdout/sealed_v3/ (LeakageGuard + audit log), evaluates the frozen plans exactly once and
        writes outputs/preregistered_blind_results.json. A second run is refused.

Blind populations (never read by any EvoAlign code before registration; only a city-level row count was queried to size them):
  P1 津冀  : Tianjin + Hebei cities, 2025-01-01 .. 2026-03-10, bachelor-accessible entry technical JDs (regional labour market)
  P2 京外全国: all non-Beijing cities, same window and population (out-of-region generalisation)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "outputs"
PROC = ROOT / "data" / "processed"
SEALED = ROOT / "data" / "blind_holdout" / "sealed_v3"
PREREG = OUT / "preregistration_blind_v3.json"
RESULT = OUT / "preregistered_blind_results.json"
DB = ROOT.parent / "work" / "modern_data" / "catalog.duckdb"
HB_CITIES = ("石家庄", "保定", "廊坊", "唐山", "秦皇岛", "邯郸", "邢台", "张家口", "承德", "沧州", "衡水")
POPULATIONS = {
    "P1_jinji": {"label": "津冀（天津+河北）2025–2026Q1 本科可及入门技术岗",
                 "where": "(work_city LIKE '%天津%' OR work_city LIKE '%河北%' OR work_city IN (" + ",".join(f"'{c}'" for c in HB_CITIES) + "))"},
    "P2_national_ex_bj": {"label": "京外全国 2025–2026Q1 本科可及入门技术岗", "where": "(work_city NOT LIKE '%北京%')"},
}
BASE_WHERE = "year IN (2025, 2026) AND broad_bachelor_accessible_entry = TRUE"
ARMS = ("B0_Freeze", "B1_SingleTrack_MILP", "B2_MultiTrack_Static", "B3_SingleShot_LLM", "B4_BiLevel_NoGuard")
HYPOTHESES = [
    {"id": "H1", "ref": "B0_Freeze", "type": "superiority", "margin": 0.0, "text": "Full 的岗位级达标率高于现状冻结（配对 95% 区间下界 > 0）"},
    {"id": "H2", "ref": "B2_MultiTrack_Static", "type": "superiority", "margin": 0.0, "text": "Full 高于多轨静态基线（下界 > 0）"},
    {"id": "H3", "ref": "B3_SingleShot_LLM", "type": "superiority", "margin": 0.0, "text": "Full 高于单轮 LLM 基线（下界 > 0）"},
    {"id": "H4", "ref": "B4_BiLevel_NoGuard", "type": "non_inferiority", "margin": -0.02, "text": "Full 不劣于关闭 GUARD 的双环（下界 > −2pp）"},
]
FROZEN = {
    "verified_master_results.json": OUT / "verified_master_results.json",
    "plan_snapshots.json": OUT / "plan_snapshots.json",
    "dynamic_kg.json": PROC / "dynamic_kg.json",
    "src/analytics/track_evaluator.py": ROOT / "src" / "analytics" / "track_evaluator.py",
    "src/analytics/gap_engine.py": ROOT / "src" / "analytics" / "gap_engine.py",
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def register() -> None:
    if PREREG.exists():
        raise SystemExit("preregistration already exists; it is immutable")
    withdrawn = sorted(OUT.glob("preregistration_blind_v3.withdrawn*.json"))
    if SEALED.exists() or RESULT.exists():
        raise SystemExit("blind data or results already exist; cannot preregister after the fact")
    master = json.loads(FROZEN["verified_master_results.json"].read_text(encoding="utf-8"))
    doc = {
        "title": "EvoAlign-OS 预注册独立盲测 v3",
        "registered_utc": now(),
        "rationale": "北京 2025 回放在研发中被查看过，只能称历史回放；本测试在冻结全部方案与评估代码后，在从未读取过的人群上一次性检验预先声明的假设。",
        "frozen_master_sha256": master["master_sha256"],
        "frozen_files_sha256": {k: sha(v) for k, v in FROZEN.items()},
        "this_script_sha256": sha(Path(__file__)),
        "data_source": str(DB.relative_to(ROOT.parent)),
        "data_table": "jobs_technical",
        "base_filter_sql": BASE_WHERE,
        "populations": POPULATIONS,
        "pre_registration_data_access": "登记前仅执行过按城市的行数统计（COUNT），未读取任何岗位文本、技能或指标",
        "programs": ["BTBU_CS_2022", "BTBU_AI_2022", "BISTU_CS_2025"],
        "plans": "plan_snapshots.json 中各专业 B0–B4 与 Full_EvoAlign_OS 的逐方向课程清单（冻结，不重新求解、不调用 LLM）",
        "primary_metric": "jd_qual_rate_70（岗位级达标率，雇主等权）",
        "analysis": "雇主聚类配对自助法，n_boot=600，seed=42，双侧 95% 百分位区间",
        "hypotheses": HYPOTHESES,
        "decision_rule": "主结论以 P1 津冀为准，P2 为稳健性；每条假设在每个专业上单独判定，不做事后删除或替换；结果无论成立与否全部报告",
        "one_shot": True,
        "withdrawn_prior_registrations": [{"file": w.name, "sha256": sha(w), "reason": json.loads(w.read_text(encoding="utf-8")).get("withdrawal_reason")} for w in withdrawn],
    }
    body = json.dumps(doc, ensure_ascii=False, sort_keys=True)
    doc["preregistration_sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
    PREREG.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    print("registered", doc["registered_utc"], doc["preregistration_sha256"][:16])


def run() -> None:
    import duckdb
    from analytics.gap_engine import extract_demand_profile
    from analytics.track_evaluator import TRACK_IDS, build_track_demand, evaluate_multitrack_plan, paired_employer_bootstrap_ci
    from security.leakage_guard import BlindAuditContext

    if RESULT.exists():
        raise SystemExit("one-shot blind test already executed; refusing to run again")
    pre = json.loads(PREREG.read_text(encoding="utf-8"))
    body = {k: v for k, v in pre.items() if k != "preregistration_sha256"}
    assert hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest() == pre["preregistration_sha256"], "preregistration tampered"
    for k, p in FROZEN.items():
        assert sha(p) == pre["frozen_files_sha256"][k], f"frozen artefact changed after registration: {k}"
    started = now()
    SEALED.mkdir(parents=True, exist_ok=False)
    con = duckdb.connect(str(DB), read_only=True)
    cols = [c[0] for c in con.execute("DESCRIBE jobs_technical").fetchall()]
    sel = ", ".join(f'"{c}"' for c in cols if c not in ("employer_name", "company_location", "work_location"))
    data = {}
    for pid_, pop in POPULATIONS.items():
        path = SEALED / f"{pid_}.parquet"
        con.execute(f"COPY (SELECT {sel} FROM jobs_technical WHERE {BASE_WHERE} AND {pop['where']}) TO '{path.as_posix()}' (FORMAT PARQUET)")
        data[pid_] = {"path": str(path.relative_to(ROOT)), "sha256": sha(path), "extracted_utc": now()}
    con.close()

    from analytics.gap_engine import load_kg
    kg = load_kg()
    snaps = json.loads(FROZEN["plan_snapshots.json"].read_text(encoding="utf-8"))
    ver = ROOT / "data" / "verification"
    ver.mkdir(parents=True, exist_ok=True)
    results = {}
    for pid_, pop in POPULATIONS.items():
        with BlindAuditContext("preregistered blind test v3 one-shot evaluation", caller="preregistered_blind_test"):
            raw = extract_demand_profile([SEALED / f"{pid_}.parquet"], kg, population="broad_entry", use_bj_shortage_calibration=False)
        td = build_track_demand(raw, kg)
        lab = {tk: i for i, tk in enumerate(TRACK_IDS)}
        np.savez_compressed(
            ver / f"features_blind_v3_{pid_}.npz",
            hits=td["hits"].astype(np.uint8), ad_weights=td["ad_weights"],
            employers=np.array([f"E{i:06d}" for i in np.unique(np.asarray(td["employers"]).astype(str), return_inverse=True)[1]]),
            job_bloom=td["job_bloom"], demand_dist=td["demand_dist"],
            track_labels=np.array([lab[t] for t in td["track_labels"]], dtype=np.int8),
            track_weight_share=np.array([td["tracks"][tk]["weight_share"] for tk in TRACK_IDS]),
            track_demand_dist=np.stack([td["tracks"][tk]["demand_dist"] for tk in TRACK_IDS]),
            track_n_ads=np.array([td["tracks"][tk]["n_ads"] for tk in TRACK_IDS]),
            n_employers=np.array(td["n_employers"]),
        )
        progs = {}
        for prog in pre["programs"]:
            arms = snaps["programs"][prog]
            evals = {arm: evaluate_multitrack_plan(s["track_curricula"], s["track_shares"], td, kg) for arm, s in arms.items()}
            cis = {ref: paired_employer_bootstrap_ci(evals, ref, td, "_per_jd_qualified", n_boot=600, seed=42)["Full_EvoAlign_OS"] for ref in ARMS}
            hyp = []
            for h in pre["hypotheses"]:
                c = cis[h["ref"]]
                hyp.append({"id": h["id"], "ref": h["ref"], "type": h["type"], "margin": h["margin"], "delta": c["delta"],
                            "ci95_lo": c["ci95_lo"], "ci95_hi": c["ci95_hi"], "supported": bool(c["ci95_lo"] > h["margin"])})
            progs[prog] = {
                "metrics": {arm: {k: ev[k] for k in ("jd_qual_rate_70", "jd_mean_readiness", "worst_track_readiness", "track_l1_overlap", "track_ot_alignment")}
                            for arm, ev in evals.items()},
                "paired_qual_ci95_full_minus": cis,
                "hypotheses": hyp,
            }
        results[pid_] = {"label": pop["label"], "n_ads": td["n_rows"], "n_employers": td["n_employers"],
                         "track_demand_shares": {tk: td["tracks"][tk]["weight_share"] for tk in TRACK_IDS}, "programs": progs}
        print(pid_, td["n_rows"], "ads", td["n_employers"], "employers", flush=True)
    supported = {pid_: sum(h["supported"] for p in r["programs"].values() for h in p["hypotheses"]) for pid_, r in results.items()}
    out = {
        "preregistration_file": PREREG.name, "preregistration_sha256": pre["preregistration_sha256"],
        "registered_utc": pre["registered_utc"], "run_started_utc": started, "finished_utc": now(),
        "frozen_master_sha256": pre["frozen_master_sha256"], "sealed_data": data,
        "hypotheses_supported": {k: f"{v}/{3 * len(HYPOTHESES)}" for k, v in supported.items()},
        "results": results,
    }
    out["result_sha256"] = hashlib.sha256(json.dumps(out, ensure_ascii=False, sort_keys=True, default=float).encode("utf-8")).hexdigest()
    RESULT.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps(out["hypotheses_supported"], ensure_ascii=False))
    for pid_, r in results.items():
        for prog, pr in r["programs"].items():
            m = pr["metrics"]
            print(pid_, prog, " ".join(f"{a[:2]}={m[a]['jd_qual_rate_70'] * 100:.1f}" for a in m),
                  " | ", " ".join(f"{h['id']}:{h['delta'] * 100:+.1f}[{h['ci95_lo'] * 100:+.1f},{h['ci95_hi'] * 100:+.1f}]{'✓' if h['supported'] else '✗'}" for h in pr["hypotheses"]))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["register", "run"])
    a = ap.parse_args()
    register() if a.cmd == "register" else run()
