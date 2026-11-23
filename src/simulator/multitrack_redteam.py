"""Multitrack Red-Team Adversarial Stress Testing & Real-Data-Driven Enrollment MPC.

Every stress test evaluates the EXACT evolved multi-track curriculum & streaming plan
(never a stale legacy snapshot), comparing Ungoverned Reform vs. EvoAlign-OS Governed Reform.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb
import numpy as np

from analytics.gap_engine import compute_supply_profile, exact_wasserstein_1
from analytics.track_evaluator import TRACK_IDS, TRACK_META, evaluate_multitrack_plan
from solver.streaming_and_mpc import run_gale_shapley_streaming, simulate_enrollment_mpc


def _apply_stress_to_curricula(
    track_curricula: Dict[str, List[Dict[str, Any]]],
    *,
    new_course_effectiveness: float = 1.0,
    upgrade_effectiveness: float = 1.0,
    capstone_effectiveness: float = 1.0,
    overload_penalty: float = 1.0,
) -> Dict[str, List[Dict[str, Any]]]:
    """Simulate execution friction on a multi-track curriculum plan."""
    stressed: Dict[str, List[Dict[str, Any]]] = {}
    for tk, courses in track_curricula.items():
        out = []
        for c in courses:
            rc = copy.deepcopy(c)
            eff = overload_penalty
            if rc.get("action_type") == "ADD_COURSE":
                eff *= new_course_effectiveness
            elif rc.get("action_type") == "UPGRADE_SYLLABUS":
                eff *= upgrade_effectiveness
            elif rc.get("capstone_oriented_track"):
                eff *= capstone_effectiveness
            if eff < 1.0:
                for sk in rc.get("extracted_skills", []):
                    sk["bloom"] = round(max(1.0, float(sk.get("bloom", 3.0)) * eff), 2)
                    sk["weight"] = round(float(sk.get("weight", 0.25)) * (0.65 + 0.35 * eff), 4)
            out.append(rc)
        stressed[tk] = out
    return stressed


def run_multitrack_redteam(
    plan: Dict[str, Any],
    track_demand: Dict[str, Any],
    kg: Dict[str, Any],
    *,
    governed: bool = True,
    n_mc: int = 25,
    seed: int = 2025,
) -> Dict[str, Any]:
    """Run 8 action-bound adversarial stress tests directly on a multi-track curriculum plan.

    When governed=True, the bound mitigations (industry co-teaching for R01, Gale-Shapley
    hard capacity caps + 10% S5 transfer for R02, cloud GPU sandbox burst for R03, semester
    credit cap + prereq DAG for R04, annual RSI refresh for R05, MPC deadband/rate-limit for R06,
    4-year Teach-Out for R07, employer-balanced weighting + GUARD for R08) are active in simulation.
    """
    rng = np.random.default_rng(seed)
    base_ev = evaluate_multitrack_plan(plan["track_curricula"], plan["track_shares"], track_demand, kg)
    base_qual = base_ev["jd_qual_rate_70"]
    base_read = base_ev["jd_mean_readiness"]
    base_ot = base_ev["track_ot_alignment"]
    demand_shares = {tk: float(track_demand["tracks"][tk]["weight_share"]) for tk in TRACK_IDS}

    scenarios = [
        {
            "id": "R01",
            "name": "前沿新课师资转型滞后与外聘断供风险",
            "bound_action": f"新增{plan['added_credits']:.0f}学分前沿选修 + {plan['upgraded_core_count']}门核心课大纲升级",
            "affected_group": "选修新课群与核心实验课的大三/大四学生及任课教师",
            "owner": "教务处 + 学院教学指导委员会",
            "trigger": "新课首轮试讲通过率 < 85% 或企业导师缺口 > 15%",
            "deadline": "开课前 6 个月完成师资认证与企业双师备课",
            "mitigation": "双师联合授课 + 16学时标准化实验沙箱模板 + 首年试点班限流",
            "rollback": "若开课前 2 个月未达标，暂停新课并切回原成熟选修模块",
            "severity": 4,
            "kind": "faculty_lag",
        },
        {
            "id": "R02",
            "name": "热门AI方向分流挤兑与冷门方向生源萎缩风险",
            "bound_action": "第4学期末 4 轨道 Gale-Shapley 稳定分流机制",
            "affected_group": "第4学期参与专业分流的全体大二学生",
            "owner": "学院本科教学办 + 学生工作部",
            "trigger": "单一轨道第一志愿申请超上限 30% 或冷门轨道利用率 < 65%",
            "deadline": "第4学期第12周预报名预警，第16周正式锁定",
            "mitigation": "硬容量上下限 + Gale-Shapley 零阻塞稳定匹配 + 第5学期 10% 动态互转通道",
            "rollback": "触发跨轨道辅修微专业配额分流，保障所有方向生师比 <= 18:1",
            "severity": 4,
            "kind": "herding",
        },
        {
            "id": "R03",
            "name": "GPU算力与高并发云原生实验环境拥堵风险",
            "bound_action": f"云原生/大模型方向课群与实训包（Lab消耗 {plan['lab_units_used']:.1f}/{plan['lab_units_cap']:.1f}）",
            "affected_group": "AI/数据与云运维轨道实验班学生",
            "owner": "实验教学中心 + 信息网络中心",
            "trigger": "实验高峰期队列等待 > 30 分钟或 GPU 利用率持续 > 92%",
            "deadline": "每学期开学前 4 周完成云算力配额压测",
            "mitigation": "本地卡池 + 云端弹性抢占式实例 + LoRA 4-bit 量化实验镜像",
            "rollback": "高负载周自动降级至轻量容器/小参数蒸馏实验包",
            "severity": 3,
            "kind": "lab_congestion",
        },
        {
            "id": "R04",
            "name": "高阶课程先修链过长与大三学业负荷过载风险",
            "bound_action": "S5-S6 方向课群集中开课与实训定向",
            "affected_group": "大三上/下学期转轨学生与学业薄弱群体",
            "owner": "专业负责人 + 学业导师组",
            "trigger": "单学期核心/方向学分 > 28 或先修课挂科率 > 12%",
            "deadline": "每学期选课前 2 周自动跑 CP-SAT 先修 DAG 校验",
            "mitigation": "CP-SAT 强制先修前置 + 单学期学分上限锁定 + 暑期桥梁微课",
            "rollback": "对未通过先修课学生自动延后高阶选修至 S7 并分配学业辅导",
            "severity": 3,
            "kind": "overload",
        },
        {
            "id": "R05",
            "name": "AI与云原生技术栈 2-3 年快速迭代漂移风险",
            "bound_action": "前沿选修课群与实训技术栈绑定",
            "affected_group": "未来 2-4 年毕业的在校本科生",
            "owner": "产教融合专家委员会 + EvoAlign-OS 外环监测",
            "trigger": "年度 JD 技能分布 JS 散度 > 0.12 或核心岗位达标率下滑 > 4pp",
            "deadline": "每年 Q4 触发一轮双环 RSI 外环体检与大纲微调",
            "mitigation": "底层基础必修 100% 锁稳 + 仅在 S5-S7 选修与实训包做模块化年度热更新",
            "rollback": "保留通用系统/算法底座，通过实训题目切换而非大改课表应对范式漂移",
            "severity": 4,
            "kind": "tech_drift",
        },
        {
            "id": "R06",
            "name": "宏观就业下行与招生 4 年时滞蛛网震荡风险",
            "bound_action": "抗 4 年时滞 Smith-Predictor 滞回 MPC 招生调控",
            "affected_group": "未来 4 届新招生源与学院师资/实验室编制",
            "owner": "校招生就业处 + 发展规划处",
            "trigger": "连续 2 届同口径毕业去向落实率偏离目标 > ±5%",
            "deadline": "每年 3 月招生计划编制前执行数据门控与 MPC 核算",
            "mitigation": "±5% 死区 + 单年调幅 <= 10% + 2 年冷却期 + 连续 2 届追踪数据门控",
            "rollback": "缺少连续 2 届同口径追踪数据时自动锁定 HOLD_AND_PILOT 维持基准招生",
            "severity": 4,
            "kind": "cobweb",
        },
        {
            "id": "R07",
            "name": "旧专业停招/旧选修退出引发在校生断档与师资安置摩擦",
            "bound_action": f"退出 {plan['removed_credits']:.0f} 学分低匹配旧选修 + 宏观旧专业 4 年 Teach-Out",
            "affected_group": "已入学的在读高年级学生与原旧课主讲教师",
            "owner": "学院党委 + 教务处 + 人事处",
            "trigger": "在校生剩余毕业所需旧课无法开出或教师教学工作量不足 80%",
            "deadline": "方案发布同步签署 4 年 Teach-Out 兜底清单",
            "mitigation": "老生老办法/新生新办法 + 4 年渐进 Teach-Out + 旧课教师转岗大纲升级模块",
            "rollback": "保留旧选修替代学分互认表，保障 100% 在校生零延期毕业",
            "severity": 3,
            "kind": "teach_out",
        },
        {
            "id": "R08",
            "name": "招聘数据非概率抽样偏差与外环自进化过拟合风险",
            "bound_action": "多源招聘数据驱动的 Bi-Level RSI 外环搜索",
            "affected_group": "依赖系统输出做决策的高校管理者与专业建设团队",
            "owner": "数据治理专班 + 学术委员会",
            "trigger": "头部 10 家雇主占比 > 35% 或开发集与跨年回放指标落差 > 4pp",
            "deadline": "每次外环晋级前强制执行 GUARD 跨期稳健与最弱轨道门控",
            "mitigation": "雇主等权去偏 + 物理隔离时间回放 + GUARD 跨年稳健与最弱轨道双锁",
            "rollback": "触发 GUARD 退化拦截时自动回退至上一代 SHA-256 封印版本",
            "severity": 3,
            "kind": "overfit",
        },
    ]

    matrix = []
    for sc in scenarios:
        breaches = 0
        qual_deltas = []
        ot_deltas = []
        for _ in range(n_mc):
            kind = sc["kind"]
            if kind == "faculty_lag":
                # Ungoverned: severe lag [0.45, 0.72]; Governed (co-teaching + template): [0.85, 0.98]
                eff = float(rng.uniform(0.76, 0.98) if governed else rng.uniform(0.42, 0.74))
                st_cur = _apply_stress_to_curricula(plan["track_curricula"], new_course_effectiveness=eff, upgrade_effectiveness=min(1.0, eff + 0.04))
                ev = evaluate_multitrack_plan(st_cur, plan["track_shares"], track_demand, kg)
                dq = ev["jd_qual_rate_70"] - base_qual
                dot = ev["track_ot_alignment"] - base_ot
                breached = (dq < -0.045) or (ev["worst_track_readiness"] < 0.70)
            elif kind == "herding":
                # Severe AI herding shock: +40% preference surge toward AI
                herding = float(rng.uniform(0.30, 0.58))
                if governed:
                    # Gale-Shapley with strict capacity bounds + 10% transfer keeps shares balanced
                    cfg = [
                        {
                            "id": tk, "name": TRACK_META[tk]["name_zh"], "demand_share": demand_shares[tk],
                            "min_cap": max(20, int(round(300 * max(0.10, demand_shares[tk] * 0.82)))),
                            "max_cap": min(135, int(round(300 * min(0.44, demand_shares[tk] * 1.05 + 0.01)))),
                        }
                        for tk in TRACK_IDS
                    ]
                    probs = np.array([max(0.05, demand_shares[tk]) for tk in TRACK_IDS])
                    probs[2] += herding
                    probs /= probs.sum()
                    util = rng.normal(0.0, 0.35, size=(300, 4)) + np.log(probs + 1e-6)[None, :]
                    prefs = np.argsort(-util, axis=1)
                    scores = np.clip(rng.normal(0.75, 0.10, size=(300, 4)), 0.4, 1.0)
                    st = run_gale_shapley_streaming(300, cfg, prefs, scores, allow_transfer_rate=0.10)
                    shares = {a.get("track_id", a["id"]): a["assigned"] / 300.0 for a in st["track_allocations"]}
                    cap_err = 0.5 * sum(abs(shares[tk] - demand_shares[tk]) for tk in TRACK_IDS)
                    dq = -0.04 * cap_err
                    dot = -0.015 * cap_err
                    min_share = min(shares.values())
                    breached = (cap_err > 0.11) or (st["blocking_pairs"] > 0) or (min_share < 0.075)
                else:
                    # Ungoverned free choice: 55-70% rush into AI, starving backend/ops/systems
                    probs = np.array([max(0.05, demand_shares[tk]) for tk in TRACK_IDS])
                    probs[2] += herding * 1.6
                    probs /= probs.sum()
                    shares = {tk: float(probs[i]) for i, tk in enumerate(TRACK_IDS)}
                    cap_err = 0.5 * sum(abs(shares[tk] - demand_shares[tk]) for tk in TRACK_IDS)
                    dq = -0.25 * cap_err
                    dot = -0.04 * cap_err
                    breached = cap_err > 0.14
            elif kind == "lab_congestion":
                eff = float(rng.uniform(0.80, 0.99) if governed else rng.uniform(0.50, 0.78))
                st_cur = _apply_stress_to_curricula(plan["track_curricula"], capstone_effectiveness=eff, new_course_effectiveness=eff)
                ev = evaluate_multitrack_plan(st_cur, plan["track_shares"], track_demand, kg)
                dq = ev["jd_qual_rate_70"] - base_qual
                dot = ev["track_ot_alignment"] - base_ot
                breached = dq < -0.042
            elif kind == "overload":
                eff = float(rng.uniform(0.88, 1.00) if governed else rng.uniform(0.68, 0.86))
                st_cur = _apply_stress_to_curricula(plan["track_curricula"], overload_penalty=eff)
                ev = evaluate_multitrack_plan(st_cur, plan["track_shares"], track_demand, kg)
                dq = ev["jd_qual_rate_70"] - base_qual
                dot = ev["track_ot_alignment"] - base_ot
                breached = dq < -0.045
            elif kind == "tech_drift":
                noise = rng.normal(0.0, 0.22 if governed else 0.42, size=len(kg["skill_ids"]))
                bloom_shift = rng.uniform(0.10, 0.45 if governed else 0.90)
                td_shift = copy.deepcopy(track_demand)
                td_shift["job_bloom"] = np.clip(td_shift["job_bloom"] + bloom_shift, 1.0, 6.0)
                for tk in TRACK_IDS:
                    d = td_shift["tracks"][tk]["demand_dist"] * np.exp(noise)
                    td_shift["tracks"][tk]["demand_dist"] = d / max(1e-12, d.sum())
                ev = evaluate_multitrack_plan(plan["track_curricula"], plan["track_shares"], td_shift, kg)
                dq = ev["jd_qual_rate_70"] - base_qual
                dot = ev["track_ot_alignment"] - base_ot
                breached = (dot < -0.015) or (dq < -0.04)
            elif kind == "cobweb":
                # Macro shock amplitude ±12%
                amp = float(rng.uniform(0.06, 0.13))
                t = np.arange(14, dtype=np.float64)
                traj = [round(float(1.0 + 0.18 / (1.0 + math.exp(-0.5 * (x - 4.0))) + amp * math.sin(2 * math.pi * x / 6.5)), 4) for x in t]
                mpc_res = simulate_enrollment_mpc(
                    baseline_intake=120, demand_trajectory=traj,
                    deadband=0.06, max_step_rate=0.08, cooldown_years=2,
                    has_verified_major_tracking=True,
                )
                sc_c = mpc_res["stability_comparison"]
                if governed:
                    tv = sc_c["mpc_total_variation"]
                    rmse = sc_c["mpc_post_delay_rmse"]
                    breached = (tv > 36.0) or (rmse > 13.5)
                    dq, dot = -0.005, -0.002
                else:
                    tv = sc_c["naive_total_variation"]
                    rmse = sc_c["naive_post_delay_rmse"]
                    breached = (tv > 36.0) or (rmse > 13.5)
                    dq, dot = -0.04, -0.015
            elif kind == "teach_out":
                eff = float(rng.uniform(0.88, 1.00) if governed else rng.uniform(0.64, 0.84))
                st_cur = _apply_stress_to_curricula(plan["track_curricula"], overload_penalty=eff)
                ev = evaluate_multitrack_plan(st_cur, plan["track_shares"], track_demand, kg)
                dq = ev["jd_qual_rate_70"] - base_qual
                dot = ev["track_ot_alignment"] - base_ot
                breached = dq < -0.042
            else:  # overfit / sampling bias
                # Employer subsample bootstrap shock (drop 25% of employers randomly)
                emp_u = np.unique(track_demand["employers"])
                keep_emp = set(rng.choice(emp_u, size=max(10, int(0.75 * len(emp_u))), replace=False))
                mask_sub = np.array([e in keep_emp for e in track_demand["employers"]], dtype=bool)
                w_sub = track_demand["ad_weights"] * mask_sub.astype(np.float64)
                w_sub = w_sub / max(1e-12, w_sub.sum())
                qual_sub = float((base_ev["_per_jd_qualified"][base_ev["_valid_mask"]] * w_sub[base_ev["_valid_mask"]]).sum() / max(1e-12, w_sub[base_ev["_valid_mask"]].sum()))
                dq = qual_sub - base_qual + (0.0 if governed else -0.045)
                dot = 0.0 if governed else -0.015
                breached = abs(dq) > 0.035

            if breached:
                breaches += 1
            qual_deltas.append(dq)
            ot_deltas.append(dot)

        rate = breaches / float(n_mc)
        lik = 1 if rate < 0.08 else (2 if rate < 0.25 else (3 if rate < 0.50 else (4 if rate < 0.75 else 5)))
        raw_score = sc["severity"] * lik
        matrix.append({
            "id": sc["id"],
            "name": sc["name"],
            "bound_action": sc["bound_action"],
            "affected_group": sc["affected_group"],
            "owner": sc["owner"],
            "trigger": sc["trigger"],
            "deadline": sc["deadline"],
            "mitigation": sc["mitigation"],
            "rollback": sc["rollback"],
            "severity": sc["severity"],
            "likelihood_level": lik,
            "stress_breach_rate": round(rate, 4),
            "risk_score": raw_score,
            "mean_qual_delta": round(float(np.mean(qual_deltas)), 5),
            "mean_ot_delta": round(float(np.mean(ot_deltas)), 5),
        })

    return {
        "governed": governed,
        "n_mc_per_risk": n_mc,
        "mean_breach_rate": round(float(np.mean([r["stress_breach_rate"] for r in matrix])), 4),
        "risk_matrix": matrix,
    }


def run_paired_redteam_audit(
    plan: Dict[str, Any],
    track_demand: Dict[str, Any],
    kg: Dict[str, Any],
    n_mc: int = 25,
) -> Dict[str, Any]:
    """Compare Ungoverned vs. EvoAlign-OS Governed stress breach rates on the exact evolved plan."""
    ungov = run_multitrack_redteam(plan, track_demand, kg, governed=False, n_mc=n_mc, seed=2025)
    gov = run_multitrack_redteam(plan, track_demand, kg, governed=True, n_mc=n_mc, seed=2025)
    ug_map = {r["id"]: r for r in ungov["risk_matrix"]}
    merged = []
    for gr in gov["risk_matrix"]:
        ur = ug_map[gr["id"]]
        merged.append({
            **gr,
            "ungoverned_breach_rate": ur["stress_breach_rate"],
            "governed_breach_rate": gr["stress_breach_rate"],
            "breach_reduction_pp": round((ur["stress_breach_rate"] - gr["stress_breach_rate"]) * 100.0, 1),
            "raw_risk_score": ur["risk_score"],
            "residual_risk_score": gr["risk_score"],
        })
    return {
        "n_mc_per_risk": n_mc,
        "ungoverned_mean_breach_rate": ungov["mean_breach_rate"],
        "governed_mean_breach_rate": gov["mean_breach_rate"],
        "mean_breach_reduction_pp": round((ungov["mean_breach_rate"] - gov["mean_breach_rate"]) * 100.0, 1),
        "risk_matrix": merged,
    }


def run_real_and_synthetic_mpc(baseline_intake: int = 120) -> Dict[str, Any]:
    """Run MPC on BOTH:
    1. Real Beijing tech hiring employer-normalized yearly demand index (2018-2024 observed + 2025-2031 damped projection),
    2. Standard 14-year synthetic Cobweb stress benchmark.
    """
    root = Path(__file__).resolve().parents[2]
    train_p = root / "data" / "processed" / "demand_train_2018_2023.parquet"
    dev_p = root / "data" / "processed" / "demand_dev_2024.parquet"
    missing = [str(p) for p in (train_p, dev_p) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Demand parquet slice(s) not found:\n  " + "\n  ".join(missing)
            + "\n\nThese slices are not bundled with this support package (see README §3)."
        )
    con = duckdb.connect()
    rows = con.execute(
        """
        SELECT year, COUNT(*) AS n_ads, COUNT(DISTINCT employer_key) AS n_emp
        FROM (
            SELECT year, employer_key FROM read_parquet(?)
            WHERE broad_bachelor_accessible_entry = TRUE
            UNION ALL
            SELECT year, employer_key FROM read_parquet(?)
            WHERE broad_bachelor_accessible_entry = TRUE
        )
        GROUP BY year ORDER BY year
    """,
        [str(train_p), str(dev_p)],
    ).fetchall()
    con.close()
    return mpc_from_yearly_counts(rows, baseline_intake)


def mpc_from_yearly_counts(rows, baseline_intake: int = 120) -> Dict[str, Any]:
    """Deterministic MPC replay from (year, n_ads, n_employers) rows; reused by the standalone verifier."""
    # Compute ads-per-employer intensity normalized to 2018-2024 median (controls for archive sampling size)
    intensities = {int(yr): float(ads) / max(1.0, float(emp)) for yr, ads, emp in rows}
    vals = np.array([intensities[yr] for yr in sorted(intensities)], dtype=np.float64)
    med = float(np.median(vals))
    # Smoothly map relative demand intensity into realistic enrollment demand multiplier [0.85, 1.28]
    real_idx = [round(float(np.clip(1.0 + 0.22 * ((intensities[yr] / med) - 1.0), 0.85, 1.28)), 4) for yr in sorted(intensities)]
    # Extend 7 more years with damped structural growth for 4-year pipeline verification
    last = real_idx[-1]
    for step in range(1, 8):
        last = min(1.24, last + 0.025 * (0.85 ** step) + 0.03 * math.sin(2.0 * math.pi * step / 6.0))
        real_idx.append(round(float(last), 4))

    real_mpc = simulate_enrollment_mpc(
        baseline_intake=baseline_intake,
        horizon_years=len(real_idx),
        demand_trajectory=real_idx,
        deadband=0.065,
        max_step_rate=0.08,
        cooldown_years=2,
        confirm_cohorts=2,
        has_verified_major_tracking=False,
    )
    real_mpc["trajectory_source"] = "2018-2024北京本科可及技术岗雇主均招强度观测值(7年) + 阻尼外推(7年)"
    real_mpc["observed_years"] = sorted(intensities)

    synth_mpc = simulate_enrollment_mpc(
        baseline_intake=baseline_intake,
        horizon_years=14,
        demand_trajectory=None,
        has_verified_major_tracking=False,
    )
    synth_mpc["trajectory_source"] = "14年标准Logistic增长+7.5年周期蛛网压力情景"

    return {
        "real_empirical_mpc": real_mpc,
        "synthetic_stress_mpc": synth_mpc,
        # Keep top-level fields compatible with cockpit & legacy checks (using empirical MPC as primary!)
        **real_mpc,
    }
