"""Standalone re-verification of every headline number in verified_master_results.json.

Re-scores the exact per-track curricula (plan_snapshots.json) on de-identified per-JD feature matrices
without any solver, LLM or network call, recomputes employer-clustered paired bootstrap CIs,
the Task 1 benchmark overlaps, the empirical MPC replay and the employment-feedback decisions,
then compares each value with the sealed master file.

Usage (anywhere, no arguments required):
    python src/experiments/verify_key_results.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

TOL = 1.2e-5
DEFAULT_ROOT = Path(__file__).resolve().parents[2]


def canonical_sha256(obj) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEFAULT_ROOT), help="Repository root (defaults to the checkout containing this script).")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = Path(args.root).resolve()
    sys.path.insert(0, str(root / "src"))
    from analytics.gap_engine import compute_supply_profile, exact_wasserstein_1  # noqa: E402
    from analytics.track_evaluator import TRACK_IDS, evaluate_multitrack_plan, paired_employer_bootstrap_ci  # noqa: E402
    from simulator.multitrack_redteam import mpc_from_yearly_counts  # noqa: E402
    from simulator.employment_feedback import run_employment_feedback  # noqa: E402

    t0 = time.perf_counter()
    ver = root / "data" / "verification"
    master = json.loads((root / "outputs" / "verified_master_results.json").read_text(encoding="utf-8"))
    snaps = json.loads((root / "outputs" / "plan_snapshots.json").read_text(encoding="utf-8"))
    kg = json.loads((ver / "kg_min.json").read_text(encoding="utf-8"))
    checks = []

    def check(group, name, got, want, tol=TOL):
        ok = (got == want) if isinstance(want, (str, bool)) or want is None else abs(float(got) - float(want)) <= tol
        checks.append({"group": group, "item": name, "recomputed": got, "sealed": want, "pass": bool(ok)})

    # 0. Seal integrity
    body = {k: v for k, v in master.items() if k != "master_sha256"}
    check("0 封印", "master_sha256", canonical_sha256(body), master["master_sha256"])
    check("0 封印", "plan_snapshots ↔ master", snaps["master_sha256"], master["master_sha256"])

    def load_td(split):
        z = np.load(ver / f"features_{split}.npz", allow_pickle=False)
        labels = np.array([TRACK_IDS[i] for i in z["track_labels"]], dtype=object)
        return {
            "n_rows": int(z["hits"].shape[0]), "n_employers": int(z["n_employers"]),
            "hits": z["hits"].astype(np.float64), "ad_weights": z["ad_weights"], "employers": z["employers"],
            "demand_dist": z["demand_dist"], "job_bloom": z["job_bloom"], "track_labels": labels,
            "tracks": {tk: {"weight_share": float(z["track_weight_share"][i]), "demand_dist": z["track_demand_dist"][i],
                            "n_ads": int(z["track_n_ads"][i])} for i, tk in enumerate(TRACK_IDS)},
        }

    tds = {"development_2024": load_td("dev_2024"), "replay_2025": load_td("replay_2025"), "replay_2026q1": load_td("replay_2026q1")}
    fields = ("jd_qual_rate_70", "jd_mean_readiness", "worst_track_readiness", "track_l1_overlap", "track_ot_alignment")

    # 1. Six-arm ablation x 3 programmes x 3 periods + paired CIs
    for pid, arms in snaps["programs"].items():
        for sp, td in tds.items():
            res = master["cases"][pid]["results"][sp]
            check("1 消融", f"{pid}/{sp}/N", td["n_rows"], res["ads"], 0)
            evals = {arm: evaluate_multitrack_plan(s["track_curricula"], s["track_shares"], td, kg) for arm, s in arms.items()}
            for arm, ev in evals.items():
                for f in fields:
                    check("1 消融", f"{pid}/{sp}/{arm}/{f}", round(ev[f], 5), res["metrics"][arm][f])
            for ref in ("B0_Freeze", "B1_SingleTrack_MILP", "B2_MultiTrack_Static", "B3_SingleShot_LLM", "B4_BiLevel_NoGuard"):
                ci = paired_employer_bootstrap_ci(evals, ref, td, "_per_jd_qualified", n_boot=600, seed=42)["Full_EvoAlign_OS"]
                want = res["paired_qual_ci95"][ref]["Full_EvoAlign_OS"]
                for k in ("delta", "ci95_lo", "ci95_hi"):
                    check("2 配对区间", f"{pid}/{sp}/Full-{ref}/{k}", ci[k], want[k])

    # 3. Natural experiment and Pareto trio on replay 2025
    td25 = tds["replay_2025"]
    ne = master["natural_experiments"]["micro_course_btbu_2022_to_2025"]
    for key, mkey in (("frozen_2022", "btbu_frozen_2022_on_replay2025"), ("real_2025", "btbu_real_2025_on_replay2025"), ("evoalign_forecast", "evoalign_pred_on_replay2025")):
        s = snaps["natural_experiment"][key]
        ev = evaluate_multitrack_plan(s["track_curricula"], s["track_shares"], td25, kg)
        check("3 自然实验", f"{key}/jd_qual_rate_70", round(ev["jd_qual_rate_70"], 5), ne[mkey]["jd_qual_rate_70"])
    for b, s in snaps["pareto_trio"].items():
        ev = evaluate_multitrack_plan(s["track_curricula"], s["track_shares"], td25, kg)
        check("3 帕累托", f"{b}/jd_qual_rate_70", round(ev["jd_qual_rate_70"], 5), master["task2_pareto_trio"][b]["replay_2025"]["jd_qual_rate_70"])

    # 4. Task 1 benchmark overlaps (skill-structure dimension)
    paths = json.loads((ver / "benchmark_pathways.json").read_text(encoding="utf-8"))
    tz = np.load(ver / "task1_targets.npz")
    C = np.asarray(kg["semantic_ground_distance"]["matrix"], dtype=np.float64)
    bench = {b["program_id"]: b for b in master["task1_gap_analysis"]["multi_university_benchmarks"]}
    for pid, p in paths.items():
        sup = compute_supply_profile(p["courses"], kg["skill_ids"])
        tgt = tz[p["role_target"]]
        check("4 任务1", f"{pid}/matched_l1_overlap", round(float(np.minimum(sup["dist"], tgt).sum()), 5), bench[pid]["matched_l1_overlap"], 2e-4)
        check("4 任务1", f"{pid}/matched_ot_alignment", round(1.0 - exact_wasserstein_1(sup["dist"], tgt, C), 5), bench[pid]["matched_ot_alignment"], 2e-4)
        check("4 任务1", f"{pid}/l1_overlap_dev2024", round(float(np.minimum(sup["dist"], tz["broad_entry"]).sum()), 5), bench[pid]["l1_overlap_dev2024"], 2e-4)

    # 5. Empirical MPC replay + employment feedback controller
    with (ver / "mpc_yearly_counts.csv").open(encoding="utf-8") as f:
        rows = [(int(r["year"]), int(r["n_ads"]), int(r["n_employers"])) for r in csv.DictReader(f)]
    mpc = mpc_from_yearly_counts(rows, 120)
    want = master["task2_mpc_enrollment"]["real_empirical_mpc"]
    for k in ("mpc_total_variation", "naive_total_variation", "oscillation_reduction_pct", "mpc_tracking_rmse", "naive_tracking_rmse"):
        check("5 招生调控", f"MPC/{k}", mpc["stability_comparison"][k], want["stability_comparison"][k], 1e-6)
    check("5 招生调控", "MPC/immediate_recommendation", mpc["immediate_recommendation"], want["immediate_recommendation"])
    efb_want = master["task2_employment_feedback"]
    efb = run_employment_feedback(120, efb_want["demand_trend_index"], ver / "employment_feedback_series.json")
    for a, b in zip(efb["decision_log"], efb_want["decision_log"]):
        check("5 招生调控", f"就业反馈/{a['decision_year']}", f"{a['state']}|{a['action']}|{a['intake_after']}", f"{b['state']}|{b['action']}|{b['intake_after']}")

    # 6. Agent-based Dream-Sim sandbox (300 student agents) recomputed from snapshots + 2025 replay features
    if master.get("task3_agent_dream_sim"):
        from simulator.agent_sandbox import run_agent_sandbox  # noqa: E402
        kg_full = json.loads((root / "data" / "processed" / "dynamic_kg.json").read_text(encoding="utf-8"))
        z = np.load(ver / "features_replay_2025.npz", allow_pickle=False)
        dem = {"demand_dist": z["demand_dist"], "hits": z["hits"].astype(np.float64), "ad_weights": z["ad_weights"], "job_bloom": z["job_bloom"]}
        for pid, want_sb in master["task3_agent_dream_sim"].items():
            got = run_agent_sandbox(snaps["programs"][pid]["B0_Freeze"], snaps["programs"][pid]["Full_EvoAlign_OS"], dem, kg_full, master["conformal_calibrator"])
            for plan in ("freeze", "full"):
                for sc in ("optimistic", "baseline", "pessimistic"):
                    check("6 多智能体沙盘", f"{pid}/{plan}/{sc}/mean_student_match", got[plan][sc]["mean_student_match"], want_sb[plan][sc]["mean_student_match"], 2e-5)

    # 7. SHA-256 signed ReplayBundle integrity
    if (root / "outputs" / "replay_bundle" / "MANIFEST.json").exists():
        from rsi.replay_bundle import verify_replay_bundle  # noqa: E402
        vb = verify_replay_bundle(root / "outputs" / "replay_bundle")
        check("7 ReplayBundle", "全部文件哈希一致", vb["files_ok"], True)
        check("7 ReplayBundle", "根哈希一致", vb["root_ok"], True)
        check("7 ReplayBundle", "绑定当前封印结果", vb["master_sha256"], master["master_sha256"])

    # 8. Preregistered one-shot independent blind test (津冀 / 京外全国 2025–2026Q1, never read before registration)
    pre_p, res_p = root / "outputs" / "preregistration_blind_v3.json", root / "outputs" / "preregistered_blind_results.json"
    if pre_p.exists() and res_p.exists():
        pre = json.loads(pre_p.read_text(encoding="utf-8"))
        res = json.loads(res_p.read_text(encoding="utf-8"))
        body = {k: v for k, v in pre.items() if k != "preregistration_sha256"}
        check("8 预注册盲测", "预注册文件哈希完整", canonical_sha256(body), pre["preregistration_sha256"])
        check("8 预注册盲测", "结果引用同一预注册", res["preregistration_sha256"], pre["preregistration_sha256"])
        extracted = min(v["extracted_utc"] for v in res["sealed_data"].values())
        check("8 预注册盲测", "登记时间早于盲测数据抽取", pre["registered_utc"] < res["run_started_utc"] <= extracted, True)
        check("8 预注册盲测", "冻结的方案快照未改动", hashlib.sha256((root / "outputs" / "plan_snapshots.json").read_bytes()).hexdigest(),
              pre["frozen_files_sha256"]["plan_snapshots.json"])
        script = root / "src" / "experiments" / "preregistered_blind_test.py"
        if script.exists():
            check("8 预注册盲测", "执行脚本与登记版本一致", hashlib.sha256(script.read_bytes()).hexdigest(), pre["this_script_sha256"])
        for pop, rp in res["results"].items():
            td = load_td(f"blind_v3_{pop}")
            check("8 预注册盲测", f"{pop}/N", td["n_rows"], rp["n_ads"], 0)
            for prog, pr in rp["programs"].items():
                arms = snaps["programs"][prog]
                evals = {arm: evaluate_multitrack_plan(s["track_curricula"], s["track_shares"], td, kg) for arm, s in arms.items()}
                for arm in arms:
                    check("8 预注册盲测", f"{pop}/{prog}/{arm}/jd_qual_rate_70", round(evals[arm]["jd_qual_rate_70"], 5), round(pr["metrics"][arm]["jd_qual_rate_70"], 5))
                for h in pr["hypotheses"]:
                    ci = paired_employer_bootstrap_ci(evals, h["ref"], td, "_per_jd_qualified", n_boot=600, seed=42)["Full_EvoAlign_OS"]
                    check("8 预注册盲测", f"{pop}/{prog}/{h['id']}/ci95_lo", ci["ci95_lo"], h["ci95_lo"])
                    check("8 预注册盲测", f"{pop}/{prog}/{h['id']}/判定", bool(ci["ci95_lo"] > h["margin"]), h["supported"])

    # 9. Rolling-origin 2025 macro addendum (partial official university announcements)
    mr_p = root / "outputs" / "macro_rolling_2025_partial.json"
    tr_p = ver / "moe_major_lifecycle_2025_partial_univ.json"
    if mr_p.exists() and tr_p.exists():
        mr = json.loads(mr_p.read_text(encoding="utf-8"))
        check("9 2025滚动复验", "真值文件哈希", hashlib.sha256(tr_p.read_bytes()).hexdigest(), mr["truth_sha256"])
        truth = json.loads(tr_p.read_text(encoding="utf-8"))
        adds = [r for r in truth["records"] if r["verified"] and r["is_cs_ai_info"] and "ADD" in r["action"]]
        def _m(p, a):
            return p[:3] in a or a[:3] in p
        for key, pred in (("evoalign", mr["evoalign_top5"]), ("persistence", mr["persistence_top5"])):
            hits = [p for p in pred if any(_m(p, r["major_name"]) for r in adds)]
            cov = round(sum(1 for r in adds if any(_m(p, r["major_name"]) for p in pred)) / max(1, len(adds)), 4)
            check("9 2025滚动复验", f"{key}/hits", "|".join(hits), "|".join(mr[f"{key}_hits"]))
            check("9 2025滚动复验", f"{key}/coverage", cov, mr[f"{key}_volume_coverage"], 1e-9)

    n_pass = sum(c["pass"] for c in checks)
    summary = {"master_sha256": master["master_sha256"], "n_checks": len(checks), "n_pass": n_pass,
               "all_pass": n_pass == len(checks), "runtime_seconds": round(time.perf_counter() - t0, 1)}
    out = Path(args.out) if args.out else root / "outputs" / "verification_report.json"
    out.write_text(json.dumps({"summary": summary, "checks": checks}, ensure_ascii=False, indent=1), encoding="utf-8")
    groups = {}
    for c in checks:
        g = groups.setdefault(c["group"], [0, 0])
        g[0] += c["pass"]
        g[1] += 1
    for g, (p, n) in groups.items():
        print(f"{g:10s} {p:4d}/{n:<4d} {'PASS' if p == n else 'FAIL'}")
    for c in checks:
        if not c["pass"]:
            print("  MISMATCH", c["item"], c["recomputed"], "!=", c["sealed"])
    print(f"TOTAL {n_pass}/{len(checks)} checks passed | SHA256 {master['master_sha256'][:16]} | {summary['runtime_seconds']}s")
    return 0 if summary["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
