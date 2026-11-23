"""Employment-feedback layer for the enrollment controller (Task 2-③).

Combines two independent feedback channels:
  * Demand channel  — Beijing bachelor-accessible tech employer intensity (MPC filtered trend, see multitrack_redteam).
  * Outcome channel — official national major-class employment outcomes (落实率 / 专业相关度 / 月收入),
    expressed RELATIVE to the 工学 average so that macro shocks common to all engineering majors cancel out.

Decision rule (explicit, auditable):
  warning(c)  : Δplacement(c) < -3pp  OR  Δrelevance(c) < -5pp
  trigger(c)  : warning for 2 consecutive cohorts (confirm_cohorts = 2)
  expand(c)   : Δplacement > +2pp AND Δincome > +5% for 2 consecutive cohorts AND demand trend > +deadband
  amplitude   : min(8%, 0.5%·|Δplacement| + 0.4%·|Δrelevance|) per decision, governance bounds [96, 156]
  cycle       : evaluated yearly after the report release (Q3); 2-year cooldown after any total-intake change
  decoupling  : demand channel up while outcome channel down => structural (skill) mismatch, not a quantity shortage:
                HOLD total intake, re-allocate track quotas and trigger the curriculum RSI outer loop first;
                cut intake only if the trigger persists 2 cohorts after the reform takes effect.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

SERIES = Path(__file__).resolve().parents[2] / "data" / "processed" / "employment_feedback_series.json"
RULES = {
    "warn_placement_gap_pp": -3.0,
    "warn_relevance_gap_pp": -5.0,
    "expand_placement_gap_pp": 2.0,
    "expand_income_gap": 0.05,
    "confirm_cohorts": 2,
    "max_step_rate": 0.08,
    "k_placement": 0.005,
    "k_relevance": 0.004,
    "cooldown_years": 2,
    "demand_deadband": 0.065,
    "governance_bounds": [96, 156],
}


def _series(records: List[Dict[str, Any]], metric: str, group: str) -> Dict[int, float]:
    return {r["cohort"]: r["value"] for r in records if r["metric"] == metric and r["group"] == group}


def run_employment_feedback(baseline_intake: int = 120, demand_trend_index: float = 1.0,
                            series_path: Optional[Path] = None) -> Dict[str, Any]:
    data = json.loads((series_path or SERIES).read_text(encoding="utf-8"))
    rec = data["records"]
    pl_cs, pl_en = _series(rec, "placement_rate", "计算机类"), _series(rec, "placement_rate", "工学")
    rl_cs, rl_en = _series(rec, "major_relevance", "计算机类"), _series(rec, "major_relevance", "工学")
    in_cs, in_en = _series(rec, "monthly_income", "计算机类"), _series(rec, "monthly_income", "工学")
    cohorts = sorted(set(rl_cs) | set(pl_cs) | set(in_cs))

    signals = []
    for c in cohorts:
        dp = round(pl_cs[c] - pl_en[c], 2) if c in pl_cs and c in pl_en else None
        dr = round(rl_cs[c] - rl_en[c], 2) if c in rl_cs and c in rl_en else None
        di = round(in_cs[c] / in_en[c] - 1.0, 4) if c in in_cs and c in in_en else None
        warn = (dp is not None and dp < RULES["warn_placement_gap_pp"]) or (dr is not None and dr < RULES["warn_relevance_gap_pp"])
        good = (dp is not None and dp > RULES["expand_placement_gap_pp"]) and (di is not None and di > RULES["expand_income_gap"])
        signals.append({
            "cohort": c,
            "cs_placement": pl_cs.get(c), "eng_placement": pl_en.get(c), "gap_placement_pp": dp,
            "cs_relevance": rl_cs.get(c), "eng_relevance": rl_en.get(c), "gap_relevance_pp": dr,
            "cs_income": in_cs.get(c), "eng_income": in_en.get(c), "gap_income_rel": di,
            "warning": bool(warn), "positive": bool(good),
        })

    demand_up = (demand_trend_index - 1.0) > RULES["demand_deadband"]
    demand_down = (1.0 - demand_trend_index) > RULES["demand_deadband"]
    log, intake, last_change = [], float(baseline_intake), None
    streak_w = streak_g = 0
    for s in signals:
        streak_w = streak_w + 1 if s["warning"] else 0
        streak_g = streak_g + 1 if s["positive"] else 0
        decision_year = s["cohort"] + 1  # report on cohort c is released in year c+1
        cooling = last_change is not None and decision_year - last_change < RULES["cooldown_years"]
        state = "GREEN"
        action, delta = "HOLD", 0.0
        if streak_w >= RULES["confirm_cohorts"]:
            state = "RED"
            mag = min(RULES["max_step_rate"], RULES["k_placement"] * abs(s["gap_placement_pp"] or 0.0) + RULES["k_relevance"] * abs(s["gap_relevance_pp"] or 0.0))
            if demand_up or not demand_down:
                action = "HOLD_TOTAL+REALLOCATE_TRACKS+TRIGGER_CURRICULUM_RSI"
            elif not cooling:
                action, delta = "REDUCE", -mag
            s["suggested_step_if_persistent"] = round(-mag, 4)
        elif s["warning"]:
            state = "AMBER"
            action = "WATCH (等待第 2 届确认)"
        elif streak_g >= RULES["confirm_cohorts"] and demand_up and not cooling:
            state = "GREEN+"
            mag = min(RULES["max_step_rate"], RULES["k_placement"] * abs(s["gap_placement_pp"] or 0.0))
            action, delta = "EXPAND", mag
        if delta:
            lo, hi = RULES["governance_bounds"]
            intake = min(hi, max(lo, round(intake * (1.0 + delta))))
            last_change = decision_year
        log.append({"decision_year": decision_year, "evidence_cohort": s["cohort"], "state": state, "action": action,
                    "intake_delta_rate": round(delta, 4), "intake_after": intake, "cooldown_active": bool(cooling)})

    latest = log[-1] if log else {}
    red_cohorts = [s for s in signals if s["warning"]]
    rel_gaps = [s["gap_relevance_pp"] for s in red_cohorts if s["gap_relevance_pp"] is not None]
    pl_gaps = [s["gap_placement_pp"] for s in signals if s["gap_placement_pp"] is not None]
    demand_txt = {"UP": "上行", "DOWN": "下行", "WITHIN_DEADBAND": "处于死区内、未下行"}
    ai_pl = _series(rec, "placement_rate", "人工智能专业").get(2025)
    ai_rl = _series(rec, "major_relevance", "人工智能专业").get(2025)
    unmet = _series(rec, "unrelated_due_to_unmet_requirements", "计算机类").get(2025)
    channel = "UP" if demand_up else ("DOWN" if demand_down else "WITHIN_DEADBAND")
    rel_lo = min(abs(g) for g in rel_gaps) if rel_gaps else 0.0
    rel_hi = max(abs(g) for g in rel_gaps) if rel_gaps else 0.0
    interpretation = (
        f"就业结果通道 {len(red_cohorts)} 届预警（计算机类相关度低于工学平均 {rel_lo:.0f}–{rel_hi:.0f}pp，"
        f"落实率差距由 {pl_gaps[0]:+.1f}pp 扩大至 {pl_gaps[-1]:+.1f}pp），北京本科可及技术岗需求通道{demand_txt[channel]}："
    )
    if demand_down:
        interpretation += "需求与结果双降，按幅度公式执行缩招。"
    else:
        interpretation += (
            "判定为结构性技能错配而非总量过剩。先保持总招生、调整方向名额并触发课程 RSI；"
            "若改革生效后连续 2 届仍为红色，再按幅度公式缩减。"
        )
    return {
        "source": data["source"],
        "scope_note": data["scope_note"],
        "rules": RULES,
        "demand_trend_index": round(demand_trend_index, 4),
        "demand_channel": "UP" if demand_up else ("DOWN" if demand_down else "WITHIN_DEADBAND"),
        "signals": signals,
        "decision_log": log,
        "current_recommendation": {
            **latest,
            "interpretation": interpretation,
        },
        "skill_mismatch_evidence": {
            "cs_relevance_2021_to_2025": [rl_cs.get(c) for c in (2021, 2022, 2023, 2024, 2025)],
            "eng_relevance_2021_to_2025": [rl_en.get(c) for c in (2021, 2022, 2023, 2024, 2025)],
            "cs_unrelated_due_to_unmet_requirements_2025_pct": unmet,
            "ai_major_2025": {"placement_rate": ai_pl, "relevance": ai_rl, "eng_placement": pl_en.get(2025), "eng_relevance": rl_en.get(2025)},
        },
    }
