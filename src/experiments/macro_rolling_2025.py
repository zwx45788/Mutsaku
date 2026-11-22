"""Rolling-origin macro natural experiment, second origin (addendum; does not modify the sealed master).

Forecast: Beijing MOE filing/approval history 2020–2024 + Beijing technical demand 2022–2024 -> rank candidate
CS/AI/information majors with the same solve_macro_major_portfolio_lifecycle used in the main experiment.
Truth: CS/AI/information majors that Beijing universities' official sites report as newly added in the 2025年度 cycle
(data/blind_holdout/moe_major_lifecycle_2025_partial_univ.json; PARTIAL — 10 universities, national attachment unavailable).
Baseline: persistence = the 5 most frequently added Beijing CS/AI/info majors in 2020–2024.
Outputs outputs/macro_rolling_2025_partial.json (+ a copy of the truth file under data/verification/).
"""
from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from analytics.gap_engine import extract_demand_profile, load_kg
from security.leakage_guard import BlindAuditContext, guard_read_path
from solver.curriculum_portfolio_solver import solve_macro_major_portfolio_lifecycle

ROOT = Path("evoalign_os")
PROC, BLIND, OUT, VER = ROOT / "data/processed", ROOT / "data/blind_holdout", ROOT / "outputs", ROOT / "data/verification"
CANDIDATES = [
    {"major_code": "080717T", "major_name": "人工智能", "core_skills": ["ml_classic", "deep_learning", "llm_posttrain", "agent_rag", "cv_multimodal", "nlp_text"], "required_fte": 15.0, "policy_priority": 0.25},
    {"major_code": "080910T", "major_name": "数据科学与大数据技术", "core_skills": ["big_data_eng", "sql_rdbms", "nosql_vector_db", "data_mining_bi", "python_eng"], "required_fte": 14.0, "policy_priority": 0.20},
    {"major_code": "140012TK", "major_name": "具身智能", "core_skills": ["robotics_embodied", "cv_multimodal", "llm_posttrain", "embedded_mcu", "control_signal"], "required_fte": 14.0, "policy_priority": 0.22},
    {"major_code": "080907T", "major_name": "智能科学与技术", "core_skills": ["ml_classic", "deep_learning", "robotics_embodied", "python_eng"], "required_fte": 14.0, "policy_priority": 0.16},
    {"major_code": "080911TK", "major_name": "网络空间安全", "core_skills": ["cyber_sec", "ai_safety_privacy", "comp_net", "os_kernel"], "required_fte": 14.0, "policy_priority": 0.18},
    {"major_code": "080710T", "major_name": "集成电路设计与集成系统", "core_skills": ["eda_ic_design", "comp_arch", "embedded_mcu", "c_cpp"], "required_fte": 16.0, "policy_priority": 0.18},
    {"major_code": "080803T", "major_name": "机器人工程", "core_skills": ["robotics_embodied", "control_signal", "embedded_mcu", "c_cpp"], "required_fte": 15.0, "policy_priority": 0.16},
    {"major_code": "071203T", "major_name": "数据科学", "core_skills": ["prob_stats", "data_mining_bi", "python_eng", "ml_classic"], "required_fte": 12.0, "policy_priority": 0.15},
    {"major_code": "080909T", "major_name": "电子与计算机工程", "core_skills": ["embedded_mcu", "comp_arch", "c_cpp", "iot_edge"], "required_fte": 14.0, "policy_priority": 0.10},
    {"major_code": "080902", "major_name": "软件工程", "core_skills": ["soft_eng_arch", "java_spring", "web_frontend", "qa_testing", "cloud_k8s"], "required_fte": 16.0, "policy_priority": 0.12},
]


def match(pred: str, actual: str) -> bool:
    return pred[:3] in actual or actual[:3] in pred


def score(pred, adds):
    act = Counter(r["major_name"] for r in adds)
    hits = [p for p in pred if any(match(p, a) for a in act)]
    cov = sum(c for a, c in act.items() if any(match(p, a) for p in pred)) / max(1, len(adds))
    return hits, round(cov, 4)


def main():
    kg = load_kg()
    hist = json.loads((PROC / "moe_major_lifecycle_2020_2023.json").read_text(encoding="utf-8"))
    with BlindAuditContext("macro rolling-origin 2025 addendum (history <=2024 -> 2025 partial)", caller="macro_rolling_2025"):
        hist += json.loads(guard_read_path(BLIND / "moe_major_lifecycle_2024_blind.json").read_text(encoding="utf-8"))
        truth_path = guard_read_path(BLIND / "moe_major_lifecycle_2025_partial_univ.json")
        truth = json.loads(truth_path.read_text(encoding="utf-8"))
    demand = extract_demand_profile([PROC / "demand_train_2018_2023.parquet", PROC / "demand_dev_2024.parquet"], kg,
                                    min_year=2022, use_bj_shortage_calibration=False)
    rank = solve_macro_major_portfolio_lifecycle([], [dict(c) for c in CANDIDATES], hist, demand, kg)
    ranked = sorted([d for d in rank["decisions"] if "viability_score" in d], key=lambda x: x["viability_score"], reverse=True)
    top5 = [d["major_name"] for d in ranked[:5]]
    persist = [n for n, _ in Counter(r["major_name"] for r in hist if r.get("is_beijing") and r.get("is_cs_ai_info") and "ADD" in r.get("action", "")).most_common(5)]
    adds = [r for r in truth["records"] if r["verified"] and r["is_cs_ai_info"] and "ADD" in r["action"]]
    e_hits, e_cov = score(top5, adds)
    p_hits, p_cov = score(persist, adds)
    VER.mkdir(parents=True, exist_ok=True)
    shutil.copy2(truth_path, VER / truth_path.name)
    out = {
        "setup": "滚动起点复验：仅用 MOE 2020–2024 北京备案/审批历史 + 2022–2024 北京技术岗需求，预测 2025 年度北京新增泛计算机专业",
        "truth_scope": truth["scope"], "truth_file": truth_path.name, "truth_sha256": hashlib.sha256(truth_path.read_bytes()).hexdigest(),
        "truth_universities": truth["universities"], "truth_cs_ai_adds": [(r["university"], r["major_name"], r["major_code"]) for r in adds],
        "history_years": sorted({r["year"] for r in hist}),
        "evoalign_ranking": [(d["major_name"], d["viability_score"]) for d in ranked],
        "evoalign_top5": top5, "evoalign_hits": e_hits, "evoalign_volume_coverage": e_cov,
        "persistence_top5": persist, "persistence_hits": p_hits, "persistence_volume_coverage": p_cov,
        "match_rule": "专业名前 3 字互相包含（与主实验 2024 检验同一规则）",
        "caveat": "真值仅覆盖 10 所在官网公布结果的北京高校，样本小，只作方向性佐证，不与 2024 全量检验等同",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    out["result_sha256"] = hashlib.sha256(json.dumps(out, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    (OUT / "macro_rolling_2025_partial.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("history_years", "evoalign_top5", "evoalign_hits", "evoalign_volume_coverage", "persistence_top5", "persistence_hits", "persistence_volume_coverage", "truth_cs_ai_adds")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
