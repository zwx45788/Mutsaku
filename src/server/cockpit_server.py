"""EvoAlign-OS v2 Decision Cockpit Backend (FastAPI, Single-Source-of-Truth + Live Multi-Track CP-SAT What-If)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from analytics.gap_engine import extract_demand_profile, load_curricula, load_kg
from analytics.pathways import expected_pathway
from analytics.track_evaluator import (
    build_track_demand,
    evaluate_multitrack_plan,
    evaluate_single_curriculum,
    strip_private_arrays,
)
from simulator.multitrack_redteam import _apply_stress_to_curricula, run_real_and_synthetic_mpc
from solver.multitrack_solver import (
    DEFAULT_CAPSTONE_PACKAGES,
    build_seed_elective_pool,
    solve_multitrack_curriculum,
)
from solver.streaming_and_mpc import simulate_enrollment_mpc

OUT_DIR = ROOT / "outputs"
WEB_DIR = ROOT / "web"
PROC_DIR = ROOT / "data" / "processed"

app = FastAPI(title="EvoAlign-OS v2 产教智对双环多轨道决策台", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_CACHE: Dict[str, Any] = {}


def _get_master() -> Dict[str, Any]:
    path = OUT_DIR / "verified_master_results.json"
    if not path.is_file():
        raise HTTPException(status_code=503, detail="verified_master_results.json not generated")
    mtime = path.stat().st_mtime
    if _CACHE.get("master_mtime") != mtime:
        _CACHE["master"] = json.loads(path.read_text(encoding="utf-8"))
        _CACHE["master_mtime"] = mtime
    return _CACHE["master"]


def _get_runtime_context() -> Dict[str, Any]:
    if "ctx" not in _CACHE:
        kg = load_kg()
        courses = load_curricula()
        train_raw = extract_demand_profile([PROC_DIR / "demand_train_2018_2023.parquet"], kg, min_year=2022, use_bj_shortage_calibration=False)
        dev_raw = extract_demand_profile([PROC_DIR / "demand_dev_2024.parquet"], kg, use_bj_shortage_calibration=False)
        _CACHE["ctx"] = {
            "kg": kg,
            "courses": courses,
            "train_td": build_track_demand(train_raw, kg),
            "dev_td": build_track_demand(dev_raw, kg),
        }
    return _CACHE["ctx"]


class WhatIfRequest(BaseModel):
    program_id: str = "BTBU_CS_2022"
    budget_mode: str = "medium"
    scenario: str = "baseline"
    multitrack: bool = True
    faculty_ramp: float = Field(default=0.94, ge=0.40, le=1.00)
    capstone_orient_ratio: float = Field(default=0.55, ge=0.0, le=0.80)


class MpcRequest(BaseModel):
    baseline_intake: int = 120
    deadband: float = 0.065
    max_step_rate: float = 0.08
    cooldown_years: int = 2
    has_verified_major_tracking: bool = False


@app.get("/api/health")
def health() -> Dict[str, Any]:
    m = _get_master()
    return {
        "status": "ok",
        "system": "EvoAlign-OS v2",
        "schema_version": m.get("schema_version"),
        "master_sha256": m.get("master_sha256", "")[:16],
        "mode": "dual_mode_unified_v2_plus_live_multitrack_cpsat",
    }


@app.get("/api/master")
def get_master() -> Dict[str, Any]:
    return _get_master()


@app.post("/api/whatif")
def run_whatif(req: WhatIfRequest) -> Dict[str, Any]:
    ctx = _get_runtime_context()
    base = [c for c in ctx["courses"] if c["program_id"] == req.program_id]
    if not base:
        base = [c for c in ctx["courses"] if c["program_id"] == "BTBU_CS_2022"]
    base, pathway_audit = expected_pathway(base)
    quota = float((pathway_audit.get("official_requirements") or {}).get("technical_elective_minimum") or 12.0)

    ev_before = strip_private_arrays(evaluate_single_curriculum(base, ctx["dev_td"], ctx["kg"]))
    sol = solve_multitrack_curriculum(
        base,
        build_seed_elective_pool(),
        ctx["train_td"],
        ctx["kg"],
        elective_quota_credits=quota,
        multitrack=req.multitrack,
        enable_capstone_orientation=(req.multitrack and req.capstone_orient_ratio > 0),
        capstone_packages=DEFAULT_CAPSTONE_PACKAGES,
        budget_mode=req.budget_mode,
        policy={"capstone_orient_ratio": req.capstone_orient_ratio},
    )
    sc_pen = {"optimistic": 1.0, "baseline": 0.96, "pessimistic": 0.84}.get(req.scenario, 0.96)
    stressed = _apply_stress_to_curricula(
        sol["track_curricula"],
        new_course_effectiveness=min(1.0, req.faculty_ramp * sc_pen),
        upgrade_effectiveness=min(1.0, (0.04 + req.faculty_ramp) * sc_pen),
        capstone_effectiveness=sc_pen,
    )
    ev_after = strip_private_arrays(evaluate_multitrack_plan(stressed, sol["track_shares"], ctx["dev_td"], ctx["kg"]))
    return {
        "program_id": req.program_id,
        "budget_mode": req.budget_mode,
        "scenario": req.scenario,
        "multitrack": req.multitrack,
        "feasible": sol["feasible"],
        "elective_quota_credits": quota,
        "added_credits": sol["added_credits"],
        "removed_credits": sol["removed_credits"],
        "upgraded_core_count": sol["upgraded_core_count"],
        "faculty_fte_used": sol["faculty_fte_used"],
        "constraint_checks": sol["constraint_checks"],
        "actions": sol["actions"],
        "track_electives": sol["track_electives"],
        "streaming_summary": sol["streaming_summary"],
        "before_dev2024": ev_before,
        "after_dev2024": ev_after,
    }


@app.post("/api/mpc")
def run_mpc(req: MpcRequest) -> Dict[str, Any]:
    return simulate_enrollment_mpc(
        baseline_intake=req.baseline_intake,
        deadband=req.deadband,
        max_step_rate=req.max_step_rate,
        cooldown_years=req.cooldown_years,
        has_verified_major_tracking=req.has_verified_major_tracking,
    )


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((WEB_DIR / "index.html").read_text(encoding="utf-8"))


if (OUT_DIR / "figures").exists():
    app.mount("/figures", StaticFiles(directory=str(OUT_DIR / "figures")), name="figures")
