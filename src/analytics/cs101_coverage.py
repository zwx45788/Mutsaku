"""计算机领域“101计划”核心课覆盖度（任务 1 的培养标准对照维度）.

Input: data/processed/cs101_core_courses.json
  {"source": ..., "url": ..., "courses": [{"name": ..., "skills": [skill_id, ...], "name_keywords": [...]}, ...]}
A programme covers a 101 core course when either (a) a course in its official catalogue matches one of the course's
name keywords, or (b) its expected-credit pathway supplies >= 1.6 credit-equivalents across the course's mapped skills.
Both evidence types are reported separately so that reviewers can audit them.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from analytics.gap_engine import compute_supply_profile
from analytics.pathways import expected_pathway

BENCH = ("PKU_CS_2024", "U03_BUAA_CS_2025", "BJTU_CS_2025", "BJTU_AI_2025", "CUP_CS_2024", "BTBU_CS_2022",
         "BTBU_AI_2022", "BTBU_SE_2022", "BTBU_AUTO_2024", "BJUT_CS_SYLLABI", "BISTU_CS_2025")
MASS_THRESHOLD = 1.6


def compute_cs101_coverage(groups: Dict[str, List[Dict[str, Any]]], kg: Dict[str, Any], spec_path: Path) -> Dict[str, Any]:
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    idx = {sid: i for i, sid in enumerate(kg["skill_ids"])}
    rows = []
    for pid in BENCH:
        if pid not in groups:
            continue
        pathway, _ = expected_pathway(groups[pid])
        sup = compute_supply_profile(pathway, kg["skill_ids"])
        names = [c.get("course_name", "") for c in groups[pid]]
        per = []
        for course in spec["courses"]:
            kw = course.get("name_keywords", [])
            name_hit = next((n for n in names if any(k in n for k in kw)), None)
            mass = float(sum(sup["mass"][idx[s]] for s in course["skills"] if s in idx))
            per.append({"course": course["name"], "catalogue_match": name_hit, "skill_mass": round(mass, 2),
                        "covered": bool(name_hit) or mass >= MASS_THRESHOLD})
        rows.append({
            "program_id": pid, "university": groups[pid][0]["university"], "major": groups[pid][0]["major"],
            "covered": sum(p["covered"] for p in per), "total": len(per),
            "catalogue_matched": sum(1 for p in per if p["catalogue_match"]),
            "missing": [p["course"] for p in per if not p["covered"]], "detail": per,
        })
    return {"source": spec.get("source"), "url": spec.get("url"), "rule": f"课程名关键词命中，或映射技能学分质量 ≥ {MASS_THRESHOLD}",
            "courses": [c["name"] for c in spec["courses"]], "programs": rows}
