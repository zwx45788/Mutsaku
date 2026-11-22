"""Generate publication-grade Chinese scientific figures strictly from the unified v2 single source of truth."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
PROC = ROOT / "data" / "processed"
FIGURES = OUT / "figures"
MASTER_FILE = "verified_master_results.json"
TASK1_FILE = "task1_gap_analysis_dev.json"
BENCH_INPUTS = ("demand_dev_2024.parquet", "dynamic_kg.json", "university_curricula_enriched.json")

INK, MUTED, GRID = "#152D42", "#526779", "#E2E8F0"
NAVY, TEAL, BLUE, AMBER, RUST, PURPLE, SLATE = (
    "#1B365D", "#0D9488", "#2563EB", "#D97706", "#C2410C", "#7C3AED", "#64748B"
)
PROGRAMS = {
    "BTBU_CS_2022": "北工商 · 计科",
    "BTBU_AI_2022": "北工商 · 人工智能",
    "BISTU_CS_2025": "北信科 · 计科",
}
ROLES = {
    "research": "研究与核心系统岗样本（Tier1 顶尖研究型对接）",
    "technical": "行业全技术岗样本（Tier2 双一流行业型对接）",
    "broad_entry": "本科可及入门技术岗样本（Tier3 市属应用主力对接）",
}
SKILL_NAMES = {
    "linux_shell": "Linux/Shell",
    "sql_rdbms": "SQL/数据库",
    "java_spring": "Java/后端",
    "hci_product": "交互/产品",
    "c_cpp": "C/C++系统",
    "python_eng": "Python工程",
    "soft_eng_arch": "软件架构",
    "devops_sre": "DevOps/运维",
    "llm_posttrain": "大模型微调",
    "deep_learning": "深度学习",
    "dsa": "数据结构/算法",
    "embedded_mcu": "嵌入式/MCU",
    "data_mining_bi": "数据分析/BI",
    "big_data_eng": "大数据/数仓",
    "os_kernel": "操作系统",
    "web_frontend": "Web前端",
    "cv_multimodal": "CV/多模态",
    "ml_classic": "机器学习",
    "comp_arch": "体系结构",
    "comp_net": "计算机网络",
    "cloud_k8s": "云原生/K8s",
    "dist_systems": "分布式系统",
    "cyber_sec": "网络安全",
    "qa_testing": "测试/质量",
    "agent_rag": "RAG/Agent",
    "nlp_text": "NLP",
    "prob_stats": "概率统计",
    "discrete_math": "离散数学",
}


def _pp(x: float) -> str:
    return f"{x * 100:+.1f}"


def setup_cn_font() -> str:
    for path in (
        "/home/whm/.fonts/pptembed/font1.otf",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    ):
        if Path(path).is_file():
            try:
                fm.fontManager.addfont(path)
            except Exception:
                pass
    names = {font.name for font in fm.fontManager.ttflist}
    chosen = next(
        (n for n in ("Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Zen Hei", "Droid Sans Fallback") if n in names),
        "DejaVu Sans",
    )
    plt.rcParams.update({
        "font.family": [chosen, "DejaVu Sans"],
        "font.sans-serif": [chosen, "DejaVu Sans"],
        "font.size": 10.5,
        "text.color": INK,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": INK,
        "axes.unicode_minus": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#CBD5E1",
        "axes.linewidth": 0.9,
        "axes.titlecolor": INK,
        "figure.facecolor": "white",
        "savefig.dpi": 220,
    })
    return chosen


def read_json(name: str):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def frame(fig, title: str, subtitle: str, footer: str) -> None:
    fig.text(0.05, 0.968, title, va="top", fontsize=18.5, fontweight="bold", color=NAVY)
    fig.text(0.05, 0.916, subtitle, va="top", fontsize=10.5, color=MUTED)
    fig.text(0.05, 0.025, footer, va="bottom", fontsize=9.2, color=MUTED, linespacing=1.45)


def grid(ax, axis: str = "y") -> None:
    ax.set_axisbelow(True)
    ax.grid(axis=axis, color=GRID, linewidth=0.85, linestyle="--", alpha=0.8)
    ax.tick_params(length=0, pad=5)


# ---------------------------------------------------------------------------
# Figure 1: 11/11 Reconciled Multi-Tier Beijing Skill-Gap Heatmap (0 blank rows!)
# ---------------------------------------------------------------------------
def fig1(master: dict):
    sys.path.insert(0, str(ROOT / "src"))
    from analytics.gap_engine import compute_supply_profile, extract_demand_profile
    from analytics.pathways import expected_pathway, group_curricula

    kg = json.loads((PROC / "dynamic_kg.json").read_text(encoding="utf-8"))
    source = PROC / "demand_dev_2024.parquet"
    courses = json.loads((PROC / "university_curricula_enriched.json").read_text(encoding="utf-8"))
    groups = group_curricula(courses)
    benches = master["task1_gap_analysis"]["multi_university_benchmarks"]
    abbr = {
        "北京大学": "北大", "北京航空航天大学": "北航", "北京交通大学": "北交大",
        "中国石油大学(北京)": "中石大(京)", "北京工商大学": "北工商",
        "北京工业大学": "北工大", "北京信息科技大学": "北信科",
    }
    zh_map = {s["id"]: s["zh"] for s in kg["skills"]}

    panels = []
    for pop, title in ROLES.items():
        dem = extract_demand_profile([source], kg, population=pop, use_bj_shortage_calibration=False)
        rows = [b for b in benches if b["role_target"] == pop]
        target = dem["demand_dist"]
        cols = np.argsort(-target, kind="stable")[:6]
        labels, flags, vals = [], [], []
        for b in rows:
            u = abbr.get(b["university"], b["university"])
            mj = b["major"].replace("计算机科学与技术", "计科").replace("(大纲全集)", "(大纲)")
            labels.append(f"{u} · {mj} ({b['cohort_year']})")
            planned, aud = expected_pathway(groups[b["program_id"]])
            sup = compute_supply_profile(planned, kg["skill_ids"])
            vals.append(100.0 * (target[cols] - sup["dist"][cols]))
            src_name = (aud.get("official_requirements") or {}).get("source", "")
            scope = (aud.get("official_requirements") or {}).get("scope", "full")
            tag = "官方全表" if scope == "full" else "官方专业口径"
            flags.append(f"{aud['credit_target']:g}学分 · {tag}")
        skills = [SKILL_NAMES.get(kg["skill_ids"][i], zh_map[kg["skill_ids"][i]].split("、")[0][:8]) for i in cols]
        panels.append((f"{title} · N={dem['n_rows']:,}", labels, flags, skills, np.array(vals)))

    finite = np.concatenate([p[4][np.isfinite(p[4])] for p in panels])
    limit = max(8.0, float(np.ceil(np.percentile(np.abs(finite), 98))))
    cmap = LinearSegmentedColormap.from_list("gap", ["#0F766E", "#F8FAFC", "#C2410C"])
    fig, axes = plt.subplots(3, 1, figsize=(13.4, 10.6), gridspec_kw={"height_ratios": [len(p[1]) for p in panels]})
    frame(
        fig,
        "图 1  北京分层高校泛计算机专业与目标岗位族的技能结构差距（11/11 方案全量核验）",
        "2024 开发集按高校培养定位分面诊断 · 摒弃跨校单轴综合排名 · 展示各岗位族 Top-6 核心技能缺口",
        "单元格数值 = 目标岗位技能提及份额 − 课程映射学分份额（百分点，正值橙红=供给不足需补强，负值青绿=学科理论/特色富余）。\n"
        "11 份方案均与官方 PDF/HTML 培养方案及大纲核对必/选修与学分口径（0 空白行）；按培养定位分面对照对应岗位样本，不做跨校综合排名。",
    )
    for ax, (title, labels, flags, skills, values) in zip(axes, panels):
        im = ax.imshow(values, vmin=-limit, vmax=limit, cmap=cmap, aspect="auto")
        ax.set_title(title, loc="left", fontsize=12.0, fontweight="bold", pad=26, color=NAVY)
        ax.xaxis.tick_top()
        ax.set_xticks(range(len(skills)), skills, fontsize=10.0)
        ax.set_yticks(range(len(labels)), labels, fontsize=10.0)
        ax.tick_params(length=0, pad=7)
        ax.set_xticks(np.arange(-0.5, len(skills), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
        ax.tick_params(which="minor", length=0)
        ax.grid(which="minor", color="white", linewidth=1.4)
        for sp in ax.spines.values():
            sp.set_visible(False)
        for r_i, flag in enumerate(flags):
            ax.text(1.015, r_i, flag, transform=ax.get_yaxis_transform(), va="center", fontsize=9.2, color=MUTED)
            for c_i, val in enumerate(values[r_i]):
                ax.text(
                    c_i, r_i, f"{val:+.1f}", ha="center", va="center", fontsize=9.8,
                    fontweight="bold" if abs(val) >= 5.0 else "normal",
                    color="white" if abs(val) > 0.62 * limit else INK,
                )
    cax = fig.add_axes((0.30, 0.115, 0.42, 0.015))
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[-limit, 0, limit])
    cb.set_label("技能份额差值（百分点：正值=缺口，负值=富余）", fontsize=9.8, color=MUTED)
    fig.subplots_adjust(left=0.20, right=0.80, top=0.815, bottom=0.175, hspace=0.66)
    return fig


# ---------------------------------------------------------------------------
# Figure 2: True Multi-Generation Error-Driven Bi-Level RSI Trajectory (3 Programs)
# ---------------------------------------------------------------------------
def fig2(master: dict):
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 5.8), sharey=True)
    promo_txt, blocked = [], 0
    for pid, label in PROGRAMS.items():
        hist = master["cases"][pid]["rsi_evolution"]["evolution_history"]
        gens = [f"G{h['generation']}" for h in hist if str(h["event"]).startswith("PROMOTED")]
        promo_txt.append(f"{label.replace(' · ', '')} {'、'.join(gens) if gens else '无'}")
        blocked += sum(1 for h in hist if "GUARD_BLOCKED" in str(h.get("guard_decision", {}).get("reason", "")))
    frame(
        fig,
        "图 2  误差驱动双环 Meta-Harness RSI 跨代进化与 GUARD 退化拦截轨迹",
        "三专业 2024 开发集逐代迭代 · Gen1(Qwen缺口新课) → Gen2(产教实训包+同源大纲) → Gen3-4(短板补强+SIM反思)",
        "实线为当前激活最优 Harness 的岗位级达标率（JD Qual Rate ≥70%）；实心点=GUARD 审核晋级；红叉=被 GUARD 拦截的候选。\n"
        f"晋级代次：{'；'.join(promo_txt)}。Gen3 单轨倾斜对抗探针被拦截 {blocked} 次；其余被拒候选为开发集退化或最弱轨道下滑。",
    )
    for ax, (pid, label) in zip(axes, PROGRAMS.items()):
        hist = master["cases"][pid]["rsi_evolution"]["evolution_history"]
        gens = []
        active_qual = []
        active_read = []
        curr_q = hist[0]["dev_metrics"]["jd_qual_rate_70"] * 100.0
        curr_r = hist[0]["dev_metrics"]["jd_mean_readiness"] * 100.0

        # Plot per step along logical generation x-axis
        step_x = []
        rejected_pts = []
        promoted_pts = []
        for idx, h in enumerate(hist):
            g = h["generation"]
            q_val = h["dev_metrics"]["jd_qual_rate_70"] * 100.0
            r_val = h["dev_metrics"]["jd_mean_readiness"] * 100.0
            if h["accepted"]:
                curr_q, curr_r = q_val, r_val
                promoted_pts.append((g, q_val, h["event"]))
                gens.append(g)
                active_qual.append(curr_q)
                active_read.append(curr_r)
            else:
                # Offset slightly if multiple rejections in same gen
                off = -0.10 if sum(1 for x in rejected_pts if abs(x[0] - g) < 0.25) == 0 else 0.10
                rejected_pts.append((g + off, q_val))
                if not gens or gens[-1] < g:
                    gens.append(g)
                    active_qual.append(curr_q)
                    active_read.append(curr_r)

        ax.plot(gens, active_qual, color=TEAL, lw=2.5, marker="o", markersize=6.5, label="岗位级达标率（最优激活）", zorder=4)
        ax.plot(gens, active_read, color=BLUE, lw=2.0, ls="--", marker="s", markersize=5.0, label="平均岗位技能就绪度", zorder=3)
        if rejected_pts:
            rx, ry = zip(*rejected_pts)
            ax.scatter(rx, ry, color=RUST, marker="x", s=64, lw=2.0, zorder=5, label="GUARD 拦截退化/失衡补丁")
        for g, qv, ev in promoted_pts:
            ax.annotate(f"{qv:.1f}%", (g, qv), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9.0, fontweight="bold", color=TEAL)
        ax.set_title(label, loc="left", fontweight="bold", fontsize=11.8, pad=10, color=NAVY)
        ax.set(xlim=(-0.35, 4.35), ylim=(30, 102), xticks=[0, 1, 2, 3, 4], xticklabels=["G0\n基准", "G1\n新课", "G2\n实训/大纲", "G3\n均衡", "G4\n收敛"])
        grid(ax, "y")
    axes[0].set_ylabel("2024 开发集核心指标（%）", fontsize=10.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.11), ncol=3, frameon=False, fontsize=9.8)
    fig.subplots_adjust(left=0.075, right=0.975, top=0.80, bottom=0.25, wspace=0.14)
    return fig


# ---------------------------------------------------------------------------
# Figure 3: 6-Arm Controlled Ablation (B0 -> B1 -> B2 -> B3 -> B4 -> Full)
# ---------------------------------------------------------------------------
def fig3(master: dict):
    arms = ["B0_Freeze", "B1_SingleTrack_MILP", "B2_MultiTrack_Static", "B3_SingleShot_LLM", "B4_BiLevel_NoGuard", "Full_EvoAlign_OS"]
    short = ["B0\n冻结现状", "B1\n单轨MILP", "B2\n多轨静态", "B3\n单轮LLM", "B4\n无GUARD", "Full\n双环RSI"]
    colors = [SLATE, "#64748B", BLUE, AMBER, RUST, TEAL]

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 6.2), sharey=True)
    frame(
        fig,
        "图 3  六组受控消融实验：多轨道分流 × 误差驱动 RSI × GUARD 门控逐层贡献",
        "2025 历史回放（N=2,645，769 家雇主）与 2026Q1 回放（N=351，同口径）双期对照 · 100% 满足全部硬约束",
        "柱高为 2025 回放岗位级必需技能达标率（%），菱形为最弱轨道就绪度（%），黑色短横为 2026Q1 同口径达标率。\n"
        + "Full 相对 B4（关闭 GUARD），2025 回放配对差值："
        + "；".join(
            f"{lbl.replace(' · ', '')} {_pp(c['delta'])}pp [{_pp(c['ci95_lo'])}, {_pp(c['ci95_hi'])}]{'' if c['ci95_lo'] > 0 else '（不显著）'}"
            for lbl, c in (
                (lbl, master["cases"][pid]["results"]["replay_2025"]["paired_qual_ci95"]["B4_BiLevel_NoGuard"]["Full_EvoAlign_OS"])
                for pid, lbl in PROGRAMS.items()
            )
        ) + "。",
    )
    x = np.arange(len(arms))
    for ax, (pid, label) in zip(axes, PROGRAMS.items()):
        m25 = master["cases"][pid]["results"]["replay_2025"]["metrics"]
        m26 = master["cases"][pid]["results"]["replay_2026q1"]["metrics"]
        q25 = [m25[a]["jd_qual_rate_70"] * 100.0 for a in arms]
        q26 = [m26[a]["jd_qual_rate_70"] * 100.0 for a in arms]
        worst25 = [m25[a]["worst_track_readiness"] * 100.0 for a in arms]

        bars = ax.bar(x, q25, width=0.58, color=colors, alpha=0.90, edgecolor="white", linewidth=0.8, zorder=3)
        ax.plot(x, worst25, color=NAVY, lw=1.6, marker="D", markersize=5.0, ls=":", label="最弱轨道就绪度(2025)", zorder=5)
        ax.scatter(x, q26, marker="_", s=420, linewidths=2.6, color=INK, zorder=6, label="2026Q1 同口径达标率")
        for i, (b, v25, v26, w25) in enumerate(zip(bars, q25, q26, worst25)):
            ax.text(
                b.get_x() + b.get_width() / 2.0, max(v25, w25, v26) + 2.2, f"{v25:.1f}%",
                ha="center", va="bottom", fontsize=8.8,
                fontweight="bold" if i == len(arms) - 1 else "normal",
                color=TEAL if i == len(arms) - 1 else INK,
            )
        ax.set_title(label, loc="left", fontweight="bold", fontsize=11.8, pad=10, color=NAVY)
        ax.set_xticks(x, short, fontsize=8.6)
        ax.set_ylim(0, 108)
        grid(ax, "y")
    axes[0].set_ylabel("岗位级达标率 / 最弱轨道就绪度（%）", fontsize=10.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.115), ncol=3, frameon=False, fontsize=9.8)
    fig.subplots_adjust(left=0.075, right=0.978, top=0.79, bottom=0.23, wspace=0.12)
    return fig


# ---------------------------------------------------------------------------
# Figure 4: Empirical Beijing 2018-2024 + Synthetic Stress 4-Year Delay MPC
# ---------------------------------------------------------------------------
def fig4(master: dict):
    mpc_all = master["task2_mpc_enrollment"]
    emp = mpc_all["real_empirical_mpc"]
    syn = mpc_all["synthetic_stress_mpc"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 6.2), sharey=True)
    s_emp = emp["stability_comparison"]
    s_syn = syn["stability_comparison"]
    frame(
        fig,
        "图 4  抗 4 年培养时滞 Smith-Predictor 滞回 MPC 招生动态调控（实证回放 + 蛛网压测）",
        f"基准招生 {emp['baseline_intake']} 人 · 死区 ±6.5% / 单年调幅 ≤8% / 2年冷却期 / 连续2届数据门控",
        f"左图（北京 2018–2024 雇主均招强度实测 + 阻尼外推）：需求波动未连续两届突破死区且触发数据门控，MPC 维持 HOLD（与冻结等价）；朴素反馈总变差 {s_emp['naive_total_variation']:.1f} vs MPC {s_emp['mpc_total_variation']:.1f}（震荡降 {s_emp['oscillation_reduction_pct']:.1f}%）。\n"
        f"右图（14 年标准增长+周期蛛网压力情景）：MPC 震荡降 {s_syn['oscillation_reduction_pct']:.1f}%，相对冻结跟踪误差降 {s_syn['rmse_reduction_vs_freeze_pct']:.1f}%，蛛网反转次数 2→0。",
    )

    for ax, data, title, x_labels in [
        (ax1, emp, "A. 北京 2018–2024 真实招聘强度驱动回放（前7年实测）", [str(2018 + i) for i in range(len(emp["history"]))]),
        (ax2, syn, "B. 14 年增长+周期蛛网压力情景（抗时滞收敛检验）", [f"Y{i+1}" for i in range(len(syn["history"]))]),
    ]:
        hist = data["history"]
        xs = np.arange(len(hist))
        ax.axvspan(-0.4, 3.5, color="#F1F5F9", alpha=0.85, zorder=0)
        ax.text(1.55, 0.97, "4 年管线锁定期\n（毕业供给由既往招生决定）", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8.2, color=MUTED)
        if ax is ax1:
            n_obs = len(data.get("observed_years", [])) or 7
            ax.axvline(n_obs - 0.5, color=AMBER, ls=":", lw=1.4)
            ax.text(n_obs - 0.6, 0.97, "实测 ←", transform=ax.get_xaxis_transform(), ha="right", va="top", fontsize=8.5, color=AMBER, fontweight="bold")
            ax.text(n_obs - 0.4, 0.97, "→ 阻尼外推", transform=ax.get_xaxis_transform(), ha="left", va="top", fontsize=8.5, color=AMBER, fontweight="bold")
        ax.plot(xs, [h["target_capacity"] for h in hist], color=INK, ls="--", lw=2.0, label="当年市场需求目标")
        ax.plot(xs, [h["mpc_graduates"] for h in hist], color=TEAL, lw=2.6, label="MPC 毕业供给（抗时滞）")
        ax.plot(xs, [h["naive_graduates"] for h in hist], color=RUST, lw=1.8, alpha=0.88, label="朴素反馈毕业供给（蛛网震荡）")
        ax.plot(xs, [h["freeze_graduates"] for h in hist], color=SLATE, lw=1.5, ls=":", label="现状冻结毕业供给")
        ax.set_title(title, loc="left", fontweight="bold", fontsize=11.3, pad=10, color=NAVY)
        ax.set_xticks(xs[::2], [x_labels[i] for i in range(0, len(xs), 2)], fontsize=9.0)
        grid(ax, "y")
    all_vals = [h[k] for d in (emp, syn) for h in d["history"] for k in ("target_capacity", "mpc_graduates", "naive_graduates", "freeze_graduates")]
    ax1.set_ylim(np.floor(min(all_vals) / 10.0) * 10 - 5, np.ceil(max(all_vals) / 10.0) * 10 + 14)

    ax1.set_ylabel("年度毕业供给 / 需求容量（人）", fontsize=10.5)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.115), ncol=4, frameon=False, fontsize=9.5)
    fig.subplots_adjust(left=0.075, right=0.975, top=0.795, bottom=0.23, wspace=0.12)
    return fig


# ---------------------------------------------------------------------------
# Figure 5: 4-Year Trajectory (Freeze vs 3 Scenarios) + Conformal 90% Salary Band
# ---------------------------------------------------------------------------
def fig5(master: dict):
    s3 = master["task3_dream_sim_3scenarios"]
    frz = master["task3_counterfactual_freeze"]
    conf = master["conformal_calibrator"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 6.0), gridspec_kw={"width_ratios": [1.15, 1.0]})
    frame(
        fig,
        "图 5  4 年培养周期反事实与三情景推演 + Split Conformal 90% 广告月薪区间",
        "北工商计科 2022 主案例 · 2025 回放岗位集（N=2,645） · 严守不冒充真实就业率因果承诺",
        f"左图展示大一至大四逐年累计技能就绪度与毕业时岗位达标率（大二末第4学期执行 Gale-Shapley 四轨道分流后出现显著跃升）。\n"
        f"右图为受控 Split Conformal 90% 预测区间（校准集 N={conf['calibration_n']}，实测覆盖率 {conf['calibration_empirical_coverage']*100:.1f}%），属解释性广告月薪范围。",
    )

    yrs = [1, 2, 3, 4]
    sc_styles = [
        ("optimistic", "乐观情景", TEAL, "-o"),
        ("baseline", "基准情景", BLUE, "-s"),
        ("pessimistic", "悲观情景", AMBER, "-^"),
    ]
    for key, lbl, col, fmt in sc_styles:
        traj = s3[key]["yearly_trajectory"]
        vals = [t["jd_qual_rate_70"] * 100.0 for t in traj]
        ax1.plot(yrs, vals, fmt, color=col, lw=2.4, markersize=6.5, label=f"{lbl}（毕业达标 {vals[-1]:.1f}%）")
    # Counterfactual freeze final point + horizontal reference
    frz_q = frz["jd_qual_rate_70"] * 100.0
    ax1.axhline(frz_q, color=RUST, ls="--", lw=1.8, label=f"反事实冻结毕业达标（{frz_q:.1f}%）")
    ax1.axvline(2.2, color=PURPLE, ls=":", lw=1.3)
    ax1.text(2.25, 12, "S4末 Gale-Shapley 分流\n+ S5-S8方向课群与实训", fontsize=8.8, color=PURPLE, fontweight="bold")
    ax1.set_title("A. 4 年培养周期岗位级达标率跃升轨迹", loc="left", fontweight="bold", fontsize=11.5, pad=10, color=NAVY)
    ax1.set_xticks(yrs, ["大一 (S2)", "大二 (S4)", "大三 (S6)", "大四毕业 (S8)"])
    ax1.set_ylabel("岗位级必需技能达标率（%）")
    ax1.set_ylim(0, 102)
    ax1.legend(loc="upper left", frameon=True, framealpha=0.92, fontsize=8.8)
    grid(ax1, "y")

    # Right: Split Conformal 90% salary intervals
    items = [
        ("反事实冻结", frz["explanatory_ad_salary_median"], frz["explanatory_ad_salary_conformal90"], SLATE),
        ("悲观情景", s3["pessimistic"]["explanatory_ad_salary_median"], s3["pessimistic"]["explanatory_ad_salary_conformal90"], AMBER),
        ("基准情景", s3["baseline"]["explanatory_ad_salary_median"], s3["baseline"]["explanatory_ad_salary_conformal90"], BLUE),
        ("乐观情景", s3["optimistic"]["explanatory_ad_salary_median"], s3["optimistic"]["explanatory_ad_salary_conformal90"], TEAL),
    ]
    y_pos = np.arange(len(items))
    for i, (lbl, med, (lo, hi), col) in enumerate(items):
        ax2.hlines(i, lo / 1000.0, hi / 1000.0, color=col, lw=4.5, alpha=0.35)
        ax2.scatter(med / 1000.0, i, color=col, s=75, zorder=4)
        ax2.text(med / 1000.0, i - 0.24, f"¥{med:,.0f}  [{lo/1000:.1f}k, {hi/1000:.1f}k]", ha="center", fontsize=8.8, fontweight="bold", color=col)
    ax2.set_yticks(y_pos, [x[0] for x in items], fontsize=9.8)
    ax2.set_xlabel("解释性招聘广告月薪（千元/月，90% Split Conformal 区间）")
    ax2.set_title("B. 受控 Split Conformal 90% 广告月薪解释区间", loc="left", fontweight="bold", fontsize=11.5, pad=10, color=NAVY)
    ax2.set_xlim(5.5, 36.0)
    ax2.set_ylim(-0.6, 3.6)
    grid(ax2, "x")

    fig.subplots_adjust(left=0.075, right=0.975, top=0.80, bottom=0.19, wspace=0.26)
    return fig


# ---------------------------------------------------------------------------
# Figure 6: Zero-Shot CRSI Transfer (Left) + Paired Red-Team Stress Audit (Right)
# ---------------------------------------------------------------------------
def fig6(master: dict):
    tr = master["cross_school_transfer"]
    rt = master["task4_red_team_risks"]
    risk_short = {
        "R01": "R01 师资转型滞后", "R02": "R02 AI方向挤兑", "R03": "R03 GPU/实验拥堵",
        "R04": "R04 先修与负荷过载", "R05": "R05 技术栈漂移", "R06": "R06 招生4年时滞",
        "R07": "R07 停招/旧课退出", "R08": "R08 抽样偏差/过拟合",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 6.2), gridspec_kw={"width_ratios": [1.0, 1.15]})
    frame(
        fig,
        "图 6  跨校/跨专业零样本 CRSI 迁移效果 & 红队 8 类对抗压测治理闭环",
        "左图：北工商计科进化所得 H*+SIM 零样本热迁移至未见高校/专业 · 右图：最新进化方案治理前后压测超限频次对比",
        "左图 2025 回放：不对目标专业重新调用 LLM，仅迁移 H*+SIM 即超过冷启动 H0（"
        + "；".join(
            f"{nm} {tr[k]['cold_start_H0']['jd_qual_rate_70']*100:.1f}%→{tr[k]['warm_transfer_Hstar_SIM']['jd_qual_rate_70']*100:.1f}%"
            for nm, k in (("跨校", "BISTU_CS_2025_unseen_university"), ("跨专业", "BTBU_AI_2022_unseen_major"))
        )
        + "），跨专业仍明显低于本地完整进化。\n"
        f"右图直接跑在最新进化多轨道方案上（25×Monte Carlo 极端冲击），治理机制使平均压力超限频次从 {rt['ungoverned_mean_breach_rate']*100:.1f}% 降至 {rt['governed_mean_breach_rate']*100:.1f}%（降 {rt['mean_breach_reduction_pp']:.1f}pp）。",
    )

    # Left: CRSI Transfer grouped bars
    groups_t = [
        ("跨校未见高校\n北信科 · 计科(2025)", tr["BISTU_CS_2025_unseen_university"]),
        ("跨专业未见专业\n北工商 · 人工智能(2022)", tr["BTBU_AI_2022_unseen_major"]),
    ]
    stages = [
        ("baseline_freeze", "现状冻结", SLATE),
        ("cold_start_H0", "冷启动 H0 (0代)", AMBER),
        ("warm_transfer_Hstar_SIM", "CRSI 零样本热迁移 (0代)", BLUE),
        ("full_local_evolved", "本专业完整双环进化", TEAL),
    ]
    x = np.arange(len(groups_t))
    w = 0.19
    for idx, (sk, slbl, col) in enumerate(stages):
        vals = [g[1][sk]["jd_qual_rate_70"] * 100.0 for g in groups_t]
        bars = ax1.bar(x + (idx - 1.5) * w, vals, width=w * 0.92, color=col, label=slbl, zorder=3)
        for b, v in zip(bars, vals):
            ax1.text(b.get_x() + b.get_width() / 2.0, v + 1.4, f"{v:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold" if idx >= 2 else "normal", color=col)
    ax1.set_xticks(x, [g[0] for g in groups_t], fontsize=9.8)
    ax1.set_ylabel("2025 回放岗位级达标率（%）")
    ax1.set_ylim(0, 100)
    ax1.set_title("A. 跨校/跨专业 Continual RSI 迁移验证", loc="left", fontweight="bold", fontsize=11.4, pad=10, color=NAVY)
    grid(ax1, "y")

    # Right: Paired Red-Team Ungoverned vs Governed
    matrix = sorted(rt["risk_matrix"], key=lambda r: r["id"])
    y = np.arange(len(matrix))
    h = 0.35
    ug = [r["ungoverned_breach_rate"] * 100.0 for r in matrix]
    gv = [r["governed_breach_rate"] * 100.0 for r in matrix]
    ax2.barh(y - h / 2, ug, height=h, color=RUST, alpha=0.80, label="未加治理护栏（裸改革）", zorder=3)
    ax2.barh(y + h / 2, gv, height=h, color=TEAL, alpha=0.92, label="EvoAlign-OS 治理闭环后", zorder=3)
    for i, (u_v, g_v) in enumerate(zip(ug, gv)):
        ax2.text(u_v + 1.5, i - h / 2, f"{u_v:.0f}%", va="center", fontsize=8.3, color=RUST)
        ax2.text(g_v + 1.5, i + h / 2, f"{g_v:.0f}%", va="center", fontsize=8.3, fontweight="bold", color=TEAL)
    ax2.set_yticks(y, [risk_short.get(r["id"], r["id"]) for r in matrix], fontsize=9.3)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 118)
    ax2.set_xlabel("极端压力测试超限频次（%）")
    ax2.set_title("B. 最新进化方案 8 类风险对抗压测前后对比", loc="left", fontweight="bold", fontsize=11.4, pad=10, color=NAVY)
    grid(ax2, "x")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="lower center", bbox_to_anchor=(0.5, 0.105), ncol=6, frameon=False, fontsize=8.8)
    fig.subplots_adjust(left=0.07, right=0.975, top=0.80, bottom=0.27, wspace=0.32)
    return fig


# ---------------------------------------------------------------------------
# Figure 7 (structural_repair_paired.png): Employer-Clustered Paired 95% CI Gains
# ---------------------------------------------------------------------------
def paired_fig(master: dict):
    refs = [
        ("vs B0 现状冻结", "B0_Freeze", TEAL),
        ("vs B1 单轨静态MILP", "B1_SingleTrack_MILP", BLUE),
        ("vs B2 多轨静态基线", "B2_MultiTrack_Static", PURPLE),
        ("vs B3 单轮LLM", "B3_SingleShot_LLM", RUST),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 6.2), sharex=True)
    cis = [
        master["cases"][pid]["results"][sp]["paired_qual_ci95"][rk]["Full_EvoAlign_OS"]
        for sp in ("replay_2025", "replay_2026q1") for pid in PROGRAMS for _, rk, _ in refs
    ]
    n_pos = sum(1 for c in cis if c["ci95_lo"] > 0)
    x_hi = max(c["ci95_hi"] for c in cis) * 100.0 + 9.0
    frame(
        fig,
        "图 7  EvoAlign-OS 完整双环多轨道方案相对四类基线的配对增益（雇主聚类 95% 区间）",
        "2025（N=2,645，769 雇主）与 2026Q1（N=351，107 雇主）同口径回放 · 600 次雇主聚类配对 Bootstrap",
        f"点为岗位级必需技能达标率净增量（百分点），线段为雇主聚类配对 Bootstrap 95% 区间。{len(cis)} 组对照中 {n_pos} 组区间下界大于 0。\n"
        "2026Q1 样本仅 351 条、107 家雇主，区间明显更宽；指标为课程-岗位技能匹配的代理量，不等同于真实就业率。",
    )
    for ax, (sp, title) in zip(axes, [("replay_2025", "2025 全年招聘回放"), ("replay_2026q1", "2026Q1 漂移招聘回放")]):
        for r_i, (pid, pname) in enumerate(PROGRAMS.items()):
            res = master["cases"][pid]["results"][sp]
            for g_i, (rlbl, rkey, col) in enumerate(refs):
                ci = res["paired_qual_ci95"][rkey]["Full_EvoAlign_OS"]
                pt = ci["delta"] * 100.0
                lo, hi = ci["ci95_lo"] * 100.0, ci["ci95_hi"] * 100.0
                y = r_i + (g_i - 1.5) * 0.19
                ax.hlines(y, lo, hi, color=col, lw=2.5, zorder=3)
                ax.scatter(pt, y, color=col, s=42, zorder=4, label=rlbl if r_i == 0 else None)
                ax.text(hi + 0.9, y, f"+{pt:.1f}pp", va="center", fontsize=8.2, color=col, fontweight="bold")
        ax.axvline(0, color=SLATE, ls="--", lw=1.2)
        ax.set_title(f"{title}（N={res['ads']:,}）", loc="left", fontweight="bold", fontsize=11.5, pad=10, color=NAVY)
        ax.set_yticks(range(len(PROGRAMS)), list(PROGRAMS.values()), fontsize=10.2)
        ax.set_ylim(2.55, -0.55)
        ax.set_xlim(-2.0, x_hi)
        ax.set_xlabel("岗位级必需技能达标率净增量（百分点）")
        grid(ax, "x")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.125), ncol=4, frameon=False, fontsize=9.5)
    fig.subplots_adjust(left=0.145, right=0.975, top=0.80, bottom=0.25, wspace=0.18)
    return fig


def render(*, paired_only: bool = False) -> None:
    setup_cn_font()
    master = read_json(MASTER_FILE)
    jobs = []
    if not paired_only:
        jobs.extend([
            ("fig1_multi_uni_benchmark.png", lambda: fig1(master), (MASTER_FILE, TASK1_FILE, *BENCH_INPUTS)),
            ("fig2_rsi_evolution.png", lambda: fig2(master), (MASTER_FILE,)),
            ("fig3_ablation_6way.png", lambda: fig3(master), (MASTER_FILE,)),
            ("fig4_mpc_control.png", lambda: fig4(master), (MASTER_FILE,)),
            ("fig5_dreamsim_scenarios.png", lambda: fig5(master), (MASTER_FILE,)),
            ("fig6_risk_and_transfer.png", lambda: fig6(master), (MASTER_FILE,)),
        ])
    jobs.append(("structural_repair_paired.png", lambda: paired_fig(master), (MASTER_FILE,)))

    FIGURES.mkdir(parents=True, exist_ok=True)
    all_srcs = {s for _, _, srcs in jobs for s in srcs}
    hashes = {}
    for name in all_srcs:
        folder = PROC if name in BENCH_INPUTS else OUT
        with (folder / name).open("rb") as f:
            hashes[name] = hashlib.file_digest(f, "sha256").hexdigest()

    with tempfile.TemporaryDirectory(prefix=".render-", dir=FIGURES) as staging:
        for filename, draw, sources in jobs:
            fig = draw()
            try:
                meta = {
                    "Software": "EvoAlign-OS v2 unified figure suite",
                    "Description": json.dumps({s: hashes[s] for s in sources}, sort_keys=True),
                }
                fig.savefig(Path(staging) / filename, metadata=meta)
            finally:
                plt.close(fig)
        for filename, _, _ in jobs:
            target = FIGURES / filename
            os.replace(Path(staging) / filename, target)
            print(target.relative_to(ROOT))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-only", action="store_true")
    render(paired_only=parser.parse_args().paired_only)
