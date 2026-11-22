"""Agent-based Dream-Sim sandbox wrapper for multi-track plans (Task 3 counterfactual + scenario evidence).

Runs simulator.dream_sim.run_dream_sim_4year (300 heterogeneous student agents with ability, prerequisite GPA and
cognitive-load caps; Gale-Shapley streaming at S4; faculty ramp; semester-by-semester skill accumulation) on
(a) the frozen current plan and (b) the recommended Full plan, under optimistic / baseline / pessimistic scenarios.
A multi-track plan is collapsed into one catalogue whose elective take-rates are the track-share-weighted
probabilities that a student takes the course; newly added electives are tagged so faculty-ramp effects apply.
"""
from __future__ import annotations

from typing import Any, Dict, List

from analytics.track_evaluator import TRACK_IDS
from simulator.dream_sim import run_dream_sim_4year

SCENARIOS = ("optimistic", "baseline", "pessimistic")
KEEP = ("scenario", "n_students", "mean_student_match", "student_match_p10_p90", "student_struggling_rate",
        "ot_alignment", "l1_overlap", "bloom_fidelity", "peak_semester_credits")


def collapse_plan(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    shares = plan["track_shares"]
    added = {a["course_name"] for a in plan.get("actions", []) if a.get("action") == "ADD_ELECTIVE"}
    out: Dict[str, Dict[str, Any]] = {}
    for tk in TRACK_IDS:
        for c in plan["track_curricula"][tk]:
            row = out.setdefault(c["course_name"], dict(c, _w=0.0))
            row["_w"] += float(shares[tk]) * float(c.get("take_rate", 1.0))
    res = []
    for c in out.values():
        c = dict(c)
        c["take_rate"] = min(1.0, c.pop("_w"))
        if c["course_name"] in added:
            c["action_type"] = "ADD_COURSE"
        res.append(c)
    return res


def run_agent_sandbox(freeze_plan: Dict[str, Any], full_plan: Dict[str, Any], demand_profile: Dict[str, Any],
                      kg: Dict[str, Any], conformal_model: Dict[str, Any], n_students: int = 300, seed: int = 42) -> Dict[str, Any]:
    out = {"method": "300 名异质学生智能体 × 8 学期逐期推演（能力、先修绩点、认知负荷上限、S4 Gale-Shapley 分流、师资爬坡）",
           "metric": "mean_student_match：学生智能体技能深度覆盖岗位技能束的加权比例（雇主等权）；p10/p90 为学生分布",
           "freeze": {}, "full": {}}
    for key, plan in (("freeze", freeze_plan), ("full", full_plan)):
        courses = collapse_plan(plan)
        for sc in SCENARIOS:
            r = run_dream_sim_4year(courses, demand_profile, kg, conformal_model, scenario=sc, n_students=n_students, seed=seed)
            out[key][sc] = {**{k: r[k] for k in KEEP},
                            "yearly_mean_student_match": [y["mean_student_match"] for y in r["yearly_trajectory"]]}
    out["full_minus_freeze_pp"] = {sc: round((out["full"][sc]["mean_student_match"] - out["freeze"][sc]["mean_student_match"]) * 100, 2)
                                   for sc in SCENARIOS}
    out["pessimistic_full_beats_optimistic_freeze"] = out["full"]["pessimistic"]["mean_student_match"] > out["freeze"]["optimistic"]["mean_student_match"]
    return out
