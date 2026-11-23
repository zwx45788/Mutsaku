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
VER_DIR = ROOT / "data" / "verification"

# The ~100 MB raw recruitment slices are not shipped with this support package, but
# data/verification/*.npz already carries the exact per-JD feature matrices (48-skill hits,
# employer-clustered weights, per-track partitions) that the evaluation and the CP-SAT
# solver consume. Building the runtime context from those keeps the cockpit fully live
# without the raw warehouse.
_VERIFICATION_PRESETS = {
    "development_2024": ("features_dev_2024.npz", "features_dev_2024.npz"),
    "replay_2025": ("features_replay_2025.npz", "features_replay_2025.npz"),
    "replay_2026q1": ("features_replay_2026q1.npz", "features_replay_2026q1.npz"),
}

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


def _load_catalogue() -> Dict[str, Any]:
    """Knowledge graph + curricula only (no demand slices). Always available in the support package."""
    if "catalogue" not in _CACHE:
        _CACHE["catalogue"] = {"kg": load_kg(), "courses": load_curricula()}
    return _CACHE["catalogue"]


def _track_demand_from_npz(path: Path, kg: Dict[str, Any]) -> Dict[str, Any]:
    """Build the same ``track_demand`` structure from a de-identified verification snapshot.

    ``features_*.npz`` stores the already-classified per-track partitions, so no raw JD
    titles are needed (``build_track_demand`` only derives them from titles when absent).
    """
    import numpy as np

    from analytics.track_evaluator import TRACK_IDS, TRACK_META

    z = np.load(path, allow_pickle=False)
    labels = np.array([TRACK_IDS[i] for i in z["track_labels"]], dtype=object)
    hits = z["hits"].astype(np.float64)
    ad_weights = np.asarray(z["ad_weights"], dtype=np.float64)
    tracks: Dict[str, Dict[str, Any]] = {}
    for i, tk in enumerate(TRACK_IDS):
        mask = labels == tk
        if mask.any():
            w = ad_weights[mask] / max(1e-12, ad_weights[mask].sum())
            prevalence = (hits[mask] * w[:, None]).sum(axis=0)
        else:
            prevalence = np.zeros(hits.shape[1])
        tracks[tk] = {
            "track_id": tk,
            "name_zh": TRACK_META[tk]["name_zh"],
            "n_ads": int(z["track_n_ads"][i]),
            "weight_share": round(float(z["track_weight_share"][i]), 5),
            "demand_dist": np.asarray(z["track_demand_dist"][i], dtype=np.float64),
            "prevalence": prevalence,
            "mask": mask,
        }
    return {
        "n_rows": int(hits.shape[0]),
        "n_employers": int(z["n_employers"]),
        "hits": hits,
        "ad_weights": ad_weights,
        "employers": np.asarray(z["employers"]),
        "demand_dist": np.asarray(z["demand_dist"], dtype=np.float64),
        "job_bloom": np.asarray(z["job_bloom"], dtype=np.float64),
        "track_labels": labels,
        "tracks": tracks,
        "source": f"data/verification/{path.name}",
    }


def _get_runtime_context() -> Dict[str, Any]:
    if "ctx" not in _CACHE:
        cat = _load_catalogue()
        kg, courses = cat["kg"], cat["courses"]
        train_p = PROC_DIR / "demand_train_2018_2023.parquet"
        dev_p = PROC_DIR / "demand_dev_2024.parquet"
        missing = [str(p) for p in (train_p, dev_p) if not p.is_file()]
        if not missing:
            train_raw = extract_demand_profile([train_p], kg, min_year=2022, use_bj_shortage_calibration=False)
            dev_raw = extract_demand_profile([dev_p], kg, use_bj_shortage_calibration=False)
            _CACHE["ctx"] = {
                "kg": kg,
                "courses": courses,
                "train_td": build_track_demand(train_raw, kg),
                "dev_td": build_track_demand(dev_raw, kg),
                "demand_source": "raw parquet slices",
            }
            return _CACHE["ctx"]

        # Fallback: the raw ~100 MB recruitment slices are not shipped, but the de-identified
        # snapshots in data/verification/ already contain the exact per-JD feature matrices
        # (48-skill hits, employer-clustered weights, per-track partitions) that evaluation
        # and the CP-SAT solver consume.
        dev_npz = VER_DIR / "features_dev_2024.npz"
        if not dev_npz.is_file():
            raise HTTPException(
                status_code=503,
                detail=(
                    "需求数据缺失：既没有 data/processed/demand_train_2018_2023.parquet 与 "
                    "demand_dev_2024.parquet（约 100 MB，见 README §3），也没有 "
                    "data/verification/features_dev_2024.npz 脱敏快照。"
                    "请用 src/ingestion/build_demand_slices.py 从官方数据生成切片，"
                    "或改用离线核验：python src/experiments/verify_key_results.py"
                ),
            )
        snapshot = _track_demand_from_npz(dev_npz, kg)
        _CACHE["ctx"] = {
            "kg": kg,
            "courses": courses,
            # Only the dev-2024 split is bundled non-blind; use it for both the fit view and
            # the evaluation view so the cockpit stays exhaustive and byte-reproducible.
            "train_td": snapshot,
            "dev_td": snapshot,
            "demand_source": f"de-identified snapshot {dev_npz.name} (raw slices not bundled)",
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
    # Validate the programme against the bundled catalogue FIRST, so an unknown id is a
    # 404 regardless of whether the (unbundled) demand slices are present.
    catalogue = _load_catalogue()
    base = [c for c in catalogue["courses"] if c["program_id"] == req.program_id]
    if not base:
        available = sorted({c["program_id"] for c in catalogue["courses"]})
        raise HTTPException(
            status_code=404,
            detail=f"Unknown program_id '{req.program_id}'. Available: {', '.join(available)}",
        )
    ctx = _get_runtime_context()
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
        "solver_status": sol.get("status"),
        "iis_conflicts": sol.get("iis_conflicts", []),
        "demand_source": ctx.get("demand_source"),
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
