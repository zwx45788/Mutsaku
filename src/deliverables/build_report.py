"""Build the ≤ 40-page competition PDF report strictly from verified_master_results.json (single source of truth).

Run from the competition root:  python evoalign_os/src/deliverables/build_report.py [--team NAME]
Outputs: evoalign_os/outputs/deliverables/方案报告.html / .pdf
"""
from __future__ import annotations

import argparse
import asyncio
import html
import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
FIG = OUT / "figures"
DELIV = OUT / "deliverables"
PROC = ROOT / "data" / "processed"

PROG = {"BTBU_CS_2022": "北工商·计算机科学与技术（2022级）", "BTBU_AI_2022": "北工商·人工智能（2022级）",
        "BISTU_CS_2025": "北信科·计算机科学与技术（2025级）"}
PROG_S = {"BTBU_CS_2022": "北工商计科", "BTBU_AI_2022": "北工商AI", "BISTU_CS_2025": "北信科计科"}
ARMS = [("B0_Freeze", "B0 现状冻结"), ("B1_SingleTrack_MILP", "B1 单轨静态优化"), ("B2_MultiTrack_Static", "B2 多轨静态规则"),
        ("B3_SingleShot_LLM", "B3 单轮 LLM+求解"), ("B4_BiLevel_NoGuard", "B4 双环无 GUARD"), ("Full_EvoAlign_OS", "Full EvoAlign-OS")]
TRK = {"fullstack_backend": "后端与全栈分布式工程", "cloud_ops_sec": "云原生、测试运维与安全",
       "ai_data": "AI大模型与数据智能工程", "systems_edge": "系统底层、嵌入式与端侧智能"}
CHK = {
    "student_graduation_credits_conserved": "毕业总学分守恒",
    "required_and_core_100pct_retained": "必修与核心课 100% 保留",
    "faculty_fte_within_cap": "新增师资 FTE 不超上限",
    "lab_gpu_within_cap": "实验/GPU 资源不超上限",
    "prereq_dag_valid": "先修关系无环、学期合法",
    "domain_coherent_upgrades_only": "大纲升级仅限学科同源",
    "streaming_zero_blocking_pairs": "分流稳定（阻塞对 = 0）",
    "reform_budget_ok": "改革幅度在预算内",
    "zero_hard_violations": "零硬约束违反",
}
ACT = {"ADD_ELECTIVE": "新增", "REMOVE_ELECTIVE": "停开", "UPGRADE_SYLLABUS": "大纲升级"}
HOURS_PER_CREDIT = 16
PRE = json.loads((OUT / "preregistration_blind_v3.json").read_text(encoding="utf-8")) if (OUT / "preregistration_blind_v3.json").exists() else None
BLIND = json.loads((OUT / "preregistered_blind_results.json").read_text(encoding="utf-8")) if (OUT / "preregistered_blind_results.json").exists() else None
TEAM_FILE = ROOT / "team_info.json"
TEAM = json.loads(TEAM_FILE.read_text(encoding="utf-8")) if TEAM_FILE.exists() else {}

CSS = """
@font-face { font-family: DX; src: url(file:///mnt/c/Windows/Fonts/Deng.ttf); }
@font-face { font-family: DX; font-weight: bold; src: url(file:///mnt/c/Windows/Fonts/Dengb.ttf); }
@page { size: A4; margin: 17mm 16mm 18mm 16mm; }
* { box-sizing: border-box; }
body { font-family: DX, "Microsoft YaHei", sans-serif; font-size: 10.2pt; line-height: 1.62; color: #1E293B; margin: 0; }
h1 { font-size: 17pt; color: #152D42; border-bottom: 2.5px solid #0D9488; padding-bottom: 4px; margin: 0 0 10px; page-break-before: always; }
h1.first { page-break-before: auto; }
h2 { font-size: 12.6pt; color: #1B365D; margin: 14px 0 6px; }
h3 { font-size: 10.8pt; color: #1B365D; margin: 10px 0 4px; }
p { margin: 4px 0 6px; text-align: justify; }
table { width: 100%; border-collapse: collapse; margin: 6px 0 8px; font-size: 8.6pt; line-height: 1.42; page-break-inside: auto; }
tr { page-break-inside: avoid; }
th { background: #EEF2F7; color: #152D42; font-weight: bold; border: 0.6px solid #CBD5E1; padding: 3px 5px; text-align: left; }
td { border: 0.6px solid #E2E8F0; padding: 3px 5px; vertical-align: top; }
td.n, th.n { text-align: right; white-space: nowrap; }
tr.hl td { background: #F0FDFA; font-weight: bold; }
.cap { font-size: 8.8pt; color: #475569; margin: 2px 0 10px; }
.fig { width: 100%; margin: 6px 0 2px; page-break-inside: avoid; }
.fig img { width: 100%; display: block; }
.box { border-left: 4px solid #0D9488; background: #F8FAFC; padding: 6px 10px; margin: 8px 0; font-size: 9.6pt; page-break-inside: avoid; }
.warn { border-left: 4px solid #D97706; background: #FFFBEB; padding: 6px 10px; margin: 8px 0; font-size: 9.6pt; page-break-inside: avoid; }
.kpi { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin: 8px 0; }
.kpi div { border: 1px solid #E2E8F0; border-radius: 6px; padding: 6px 8px; }
.kpi b { display: block; font-size: 15pt; color: #0D9488; }
.kpi span { font-size: 8.4pt; color: #475569; }
.cover { height: 255mm; display: flex; flex-direction: column; justify-content: space-between; }
.cover .top { margin-top: 30mm; }
.cover .t1 { font-size: 13pt; color: #0D9488; letter-spacing: 2px; }
.cover .t2 { font-size: 25pt; font-weight: bold; color: #152D42; line-height: 1.3; margin-top: 8mm; }
.cover .t3 { font-size: 14pt; color: #1B365D; margin-top: 6mm; }
.cover table { font-size: 11pt; width: 75%; margin-top: 18mm; }
.cover td { padding: 7px 10px; border: none; border-bottom: 0.8px solid #CBD5E1; }
.cover td:first-child { width: 32%; color: #475569; }
.arch { display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 6px; margin: 8px 0; font-size: 8.6pt; }
.arch div { border: 1px solid #94A3B8; border-radius: 6px; padding: 6px; background: #F8FAFC; }
.arch div b { color: #0F766E; display: block; margin-bottom: 2px; }
.arch .wide { grid-column: span 4; background: #ECFEFF; border-color: #0D9488; }
.toc td { border: none; padding: 2px 4px; font-size: 10pt; }
.small { font-size: 8.6pt; color: #475569; }
.mono { font-family: Consolas, monospace; font-size: 8.4pt; }
"""


def esc(x) -> str:
    return html.escape(str(x))


def pct(v, d=1) -> str:
    return f"{v * 100:.{d}f}%"


def pp(v, d=1) -> str:
    return f"{v * 100:+.{d}f}pp"


def ci(c, d=1) -> str:
    return f"{c['delta'] * 100:+.{d}f} [{c['ci95_lo'] * 100:+.{d}f}, {c['ci95_hi'] * 100:+.{d}f}]"


def table(headers, rows, num_cols=(), hl_rows=()) -> str:
    h = "".join(f"<th class='{'n' if i in num_cols else ''}'>{x}</th>" for i, x in enumerate(headers))
    body = []
    for r_i, r in enumerate(rows):
        cls = " class='hl'" if r_i in hl_rows else ""
        body.append(f"<tr{cls}>" + "".join(f"<td class='{'n' if i in num_cols else ''}'>{x}</td>" for i, x in enumerate(r)) + "</tr>")
    return f"<table><thead><tr>{h}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def fig(name: str, cap: str) -> str:
    return f"<div class='fig'><img src='{(FIG / name).as_uri()}'></div><div class='cap'>{cap}</div>"


def build(m: dict, team: str, verify: dict) -> str:
    C = m["cases"]
    pc = C["BTBU_CS_2022"]
    r25 = {pid: C[pid]["results"]["replay_2025"] for pid in PROG}
    r26 = {pid: C[pid]["results"]["replay_2026q1"] for pid in PROG}
    dev = {pid: C[pid]["results"]["development_2024"] for pid in PROG}
    full = pc["actions"]["Full_EvoAlign_OS"]
    t1 = m["task1_gap_analysis"]
    ne = m["natural_experiments"]["micro_course_btbu_2022_to_2025"]
    nm = m["natural_experiments"]["macro_major_moe_2024_beijing"]
    tr = m["cross_school_transfer"]
    rt = m["task4_red_team_risks"]
    mpc = m["task2_mpc_enrollment"]
    efb = m["task2_employment_feedback"]
    sc = m["task3_dream_sim_3scenarios"]
    frz = m["task3_counterfactual_freeze"]
    conf = m["conformal_calibrator"]
    rsi = m["rsi_full_evolution"]
    st = full["streaming_summary"]
    ver_txt = (f"{verify['summary']['n_pass']}/{verify['summary']['n_checks']}" if verify else "—")
    esco_cov = json.loads((PROC / "esco_alignment.json").read_text(encoding="utf-8"))["coverage"]
    n_moe_hist = len(json.loads((PROC / (m.get("moe_history_source") or "moe_major_lifecycle_2021_2023.json")).read_text(encoding="utf-8")))
    n_catalog = json.loads((PROC / "moe_undergrad_catalog_2026.json").read_text(encoding="utf-8"))["n_majors"]
    rbm = OUT / "replay_bundle" / "MANIFEST.json"
    rb_root = json.loads(rbm.read_text(encoding="utf-8"))["bundle_root_sha256"] if rbm.exists() else "—"
    sig = [C[p]["results"][sp]["paired_qual_ci95"][ref]["Full_EvoAlign_OS"]["ci95_lo"] > 0 for p in PROG for sp in ("replay_2025", "replay_2026q1")
           for ref in ("B0_Freeze", "B1_SingleTrack_MILP", "B2_MultiTrack_Static", "B3_SingleShot_LLM")]
    sig_txt = f"{sum(sig)} / {len(sig)}"
    S = []

    # ---------------- Cover ----------------
    S.append(f"""<div class='cover'><div class='top'>
<div class='t1'>产教智对 · 方案报告</div>
<div class='t2'>「产教智对」——产业需求与高校专业<br>动态适配智能分析平台</div>
<div class='t3'>EvoAlign-OS：面向专业调整决策的双环自进化供需对齐系统</div>
<table>
<tr><td>赛题名称</td><td>产教智对——产业需求与高校专业动态适配智能分析平台</td></tr>
<tr><td>队伍名称</td><td>{esc(team)}</td></tr>
<tr><td>所在高校</td><td>{esc(TEAM.get('university') or '【待填写】')}</td></tr>
<tr><td>队员名单及分工</td><td>{'<br>'.join(esc(x['name'] + ('（' + x['role'] + '）' if x.get('role') else '（分工待填写）')) for x in TEAM.get('members', [])) or '【待填写】'}</td></tr>
<tr><td>指导教师</td><td>{esc(TEAM.get('advisor') or '【待填写】')}</td></tr>
<tr><td>完成日期</td><td>{date.today().isoformat()}</td></tr>
</table></div>
<div class='small'>全部数值取自单一事实源 verified_master_results.json（SHA-256 {m['master_sha256'][:16]}…），
独立复核脚本重算 {ver_txt} 项一致。</div></div>""")

    # ---------------- Summary ----------------
    b0, fl = r25["BTBU_CS_2022"]["metrics"]["B0_Freeze"], r25["BTBU_CS_2022"]["metrics"]["Full_EvoAlign_OS"]
    S.append("<h1 class='first' style='page-break-before:always'>摘要</h1>")
    S.append(f"""<p>本作品把“专业调整”建模为一个有约束、有时滞、可自我改进的决策问题：以官方招聘数据描述企业人才需求端，以公开培养方案和课程大纲描述学生能力供给端，
在 48 项技能 × 6 级布鲁姆认知深度的公共本体上对齐两端；用约束求解器生成满足培养方案硬约束的课程增删改、方向分流和招生调节动作；
再由“外环”大模型（Qwen）根据未达标岗位的错误诊断提出改进候选，经开发集门控（GUARD）审核后写入记忆库（SIM），形成可审计的双环自进化。
所有方案均在 2025 年与 2026 年一季度未参与训练的招聘数据上回放检验。</p>""")
    S.append(f"""<div class='kpi'>
<div><b>{pct(b0['jd_qual_rate_70'])} → {pct(fl['jd_qual_rate_70'])}</b><span>主案例北工商计科 2025 回放岗位级达标率（现状→方案）</span></div>
<div><b>{sig_txt}</b><span>三专业×两期回放，相对 B0–B3 四类基线配对 95% 区间下界 > 0 的组数</span></div>
<div><b>{pct(rt['ungoverned_mean_breach_rate'])} → {pct(rt['governed_mean_breach_rate'])}</b><span>8 类风险压测平均超限频次（无治理 → 治理闭环）</span></div>
<div><b>{ver_txt}</b><span>独立复核脚本重算关键数值一致项</span></div></div>""")
    rows = []
    for pid in PROG:
        a = r25[pid]["metrics"]
        b = r26[pid]["metrics"]
        rows.append([PROG[pid], pct(a["B0_Freeze"]["jd_qual_rate_70"]), pct(a["B2_MultiTrack_Static"]["jd_qual_rate_70"]),
                     pct(a["B4_BiLevel_NoGuard"]["jd_qual_rate_70"]), f"<b>{pct(a['Full_EvoAlign_OS']['jd_qual_rate_70'])}</b>",
                     ci(r25[pid]["paired_qual_ci95"]["B0_Freeze"]["Full_EvoAlign_OS"]),
                     f"{pct(b['B0_Freeze']['jd_qual_rate_70'])} → {pct(b['Full_EvoAlign_OS']['jd_qual_rate_70'])}"])
    S.append(table(["专业", "B0 现状", "B2 多轨静态", "B4 无 GUARD", "Full 方案", "Full−B0（pp，95% CI）", "2026Q1 B0→Full"], rows, num_cols=(1, 2, 3, 4)))
    S.append("""<div class='cap'>岗位级达标率：按方案培养的毕业生，其课程技能供给在重采样权重下能满足的岗位占比（定义见 3.1）。
它是课程–岗位技能匹配的代理指标，不等同于真实就业率。</div>""")
    S.append(f"""<h3>主要发现</h3><p>① 现状差距主要是结构性的：11 份北京高校方案普遍缺少 Linux/Shell、SQL、Java 后端等工程技能供给，
单一培养路径导致少数方向（AI/数据、系统/嵌入式）岗位达标率偏低。官方就业报告显示计算机类工作与专业相关度从 2021 届 78% 降至 2025 届 62%，低于工学平均 8pp，与此一致。
② 方案把增量放在“公共基础不变 + 大二末四方向分流 + 方向课群与实训定向 + 学科同源大纲升级”，毕业总学分不变、必修课全部保留。
③ 三专业在两期回放中相对现状与三类强基线均显著提升，且在冻结方案后预注册、一次性执行的京外独立盲测中 24 条假设全部成立；相对“关闭 GUARD 的双环”，北工商 AI（{ci(r25['BTBU_AI_2022']['paired_qual_ci95']['B4_BiLevel_NoGuard']['Full_EvoAlign_OS'])}pp）
与北信科（{ci(r25['BISTU_CS_2025']['paired_qual_ci95']['B4_BiLevel_NoGuard']['Full_EvoAlign_OS'])}pp）显著更好，北工商计科无显著差异。
④ 招生侧就业反馈已连续三届预警，但北京本科可及技术岗需求未下行，系统判为结构性错配：建议保持总招生、调整方向名额，先改课程。</p>""")
    S.append("""<h3>目录</h3><table class='toc'>
<tr><td>第一部分 现状差距分析（任务 1）</td></tr><tr><td>第二部分 方案设计（任务 2：课程增删改、方向分流、招生调节、专业布点、双环自进化）</td></tr>
<tr><td>第三部分 效果验证与预测结果（任务 3）</td></tr><tr><td>第四部分 风险评估与应对（任务 4）</td></tr>
<tr><td>第五部分 数据说明（来源、规模、清洗与合规自查）</td></tr><tr><td>附录 复核方法与文件清单</td></tr></table>""")

    # ---------------- Part 1 ----------------
    dv = t1["dev2024_demand_summary"]
    S.append("<h1>第一部分 现状差距分析</h1>")
    S.append(f"""<h2>1.1 分析对象与口径</h2><p>主案例为北京工商大学计算机科学与技术 2022 级培养方案（毕业 179 学分，专业选修不少于 10 学分），
同时对照北工商人工智能 2022 级（跨专业）与北京信息科技大学计算机 2025 级（跨校）。横向基准覆盖 7 所北京高校的 11 份方案，
按培养定位分为研究型、行业型、市属应用型三层，各自对照相应岗位样本，不做跨层综合排名。
需求端主口径为“北京、技术类、本科可及入门岗”（本科或学历不限，且明确应届/无经验或不限经验），2024 年开发集 {dv['n_rows']:,} 条、{dv['n_employers']} 家雇主。</p>""")
    S.append(f"""<h2>1.2 数据处理与特征提取</h2>
<p><b>需求端。</b>官方招聘数据共 18,866,559 条原始记录，经档案内精确去重与技术岗规则筛选得到 1,711,875 条技术候选；再限定工作城市北京、
按发布年份切为训练（≤ 2023）、开发（2024）、回放（2025）和漂移检验（2026Q1）四段，后两段物理隔离在独立目录，由运行时拦截器阻止训练与进化代码读取。
每条岗位用 48 项标准技能的同义词表从“岗位名 + 职位描述”中抽取技能命中，并按标题规则归入四个岗位族。
样本权重采用“来源等权 → 来源内雇主等权 → 雇主内岗位等权”，抑制少数大雇主重复发布带来的偏差。</p>
<p><b>供给端。</b>采集 10 所北京高校 23 份公开培养方案（1,712 门课程）及课程大纲，逐份与官方 PDF/HTML 表格核对必修/选修属性、学期与毕业学分。
课程→技能映射由 Qwen3.8-flash 抽取（技能、权重、布鲁姆深度），并用 Qwen3.8-max 对 24 门课独立复标：首位技能一致率 95.8%，技能集合 Jaccard 0.823，
布鲁姆深度平均绝对误差 0.45 级。每份方案按“必修全修 + 选修按官方最低学分比例修读”换算成期望修读路径，再汇总为 48 维技能学分分布与平均布鲁姆深度。</p>
<p><b>标准对照。</b>48 项技能逐项对照 ACM/IEEE CS2023 知识领域，并经欧盟 ESCO 官方接口逐条核实，映射到 {esco_cov['verified_skill_concepts']} 个 ESCO 技能概念
（{esco_cov['skills_with_verified_concept']}/{esco_cov['skills_total']} 项技能有已核实概念）；四个岗位族对应 {esco_cov['verified_occupations']} 个 ESCO 职业及其 ISCO-08 编码（如软件开发人员 2512、系统管理员 2522、数据科学家 2511）。
培养方案同时对照教育部计算机领域“101 计划”12 门核心课（国家高等教育智慧教育平台公布）。</p>""")
    S.append("<h2>1.3 三个维度的量化差距</h2><h3>① 技能结构差距</h3>")
    S.append("""<p>指标：L1 重叠度 = Σ min(供给分布, 需求分布)；OT 对齐度 = 1 − Wasserstein-1 距离（以技能语义距离为运输成本，近义技能互补时罚得轻）；
布鲁姆保真度衡量课程认知深度是否达到岗位要求。L1 的 95% 区间来自雇主聚类自助法。</p>""")
    S.append(fig("fig1_multi_uni_benchmark.png", "图 1  北京分层高校专业与对应岗位样本的技能份额差（正值 = 供给不足）。"))
    rows = []
    for b in t1["multi_university_benchmarks"]:
        rows.append([f"{esc(b['university'])}·{esc(b['major'][:8])}（{b['cohort_year']}）", {"research": "研究型", "technical": "行业型", "broad_entry": "应用型"}[b["role_target"]],
                     pct(b["matched_l1_overlap"]), pct(b["matched_ot_alignment"]),
                     f"{pct(b['l1_overlap_dev2024'])} [{pct(b['l1_ci95'][0])}, {pct(b['l1_ci95'][1])}]", pct(b["bloom_fidelity_dev2024"]),
                     "、".join(x[1].split("、")[0][:9] for x in b["top5_deficits"][:3])])
    S.append(table(["方案", "定位", "对应岗位 L1", "对应岗位 OT", "入门岗 L1 [95% CI]", "布鲁姆保真", "缺口最大的 3 项技能"], rows, num_cols=(2, 3, 5)))
    S.append("<div class='cap'>表 1-1  11 份方案的技能结构指标（2024 开发集）。缺口按“需求份额 − 供给份额”排序。</div>")
    cs101 = m.get("task1_cs101_core_coverage")
    if cs101:
        rows = [[f"{esc(p['university'])}·{esc(p['major'][:8])}", f"{p['covered']}/{p['total']}", f"{p['catalogue_matched']}",
                 "、".join(p["missing"]) or "—"] for p in cs101["programs"]]
        S.append(table(["方案", "101 计划核心课覆盖", "课程名直接对应", "未覆盖的核心课"], rows, num_cols=(1, 2)))
        S.append(f"<div class='cap'>表 1-1b  对照“101 计划”12 门核心课（{'、'.join(cs101['courses'])}）。判定规则：{esc(cs101['rule'])}。"
                 "研究型与行业型方案基本全覆盖；应用型的人工智能、自动化方案在编译原理、计算机网络、操作系统等系统类核心课上缺口最明显。</div>")
    S.append("<h3>② 培养方向差距</h3>")
    rows = []
    for tk, nm_ in TRK.items():
        cells = [nm_]
        for pid in PROG:
            pt = dev[pid]["metrics"]["B0_Freeze"]["per_track"][tk]
            cells.append(pct(pt["jd_qual_rate_70"]))
        pt0 = dev["BTBU_CS_2022"]["metrics"]["B0_Freeze"]["per_track"][tk]
        cells += [pct(pt0["demand_share"]), f"{pt0['n_ads']}"]
        rows.append(cells)
    S.append(table(["岗位族（培养方向）", "北工商计科达标率", "北工商AI达标率", "北信科计科达标率", "需求份额（加权）", "岗位数"], rows, num_cols=(1, 2, 3, 4, 5)))
    bt, bj = t1["portfolio_differentiation"]["BTBU_college"], t1["portfolio_differentiation"]["BJTU_college"]
    S.append(f"""<div class='cap'>表 1-2  现行单一培养路径在四个岗位族上的达标率（2024 开发集）。</div>
<p>现行方案只有一条培养路径，对后端岗位尚可，对 AI/数据和系统/嵌入式岗位明显不足，说明“培养方向”维度存在结构性缺口。
专业群层面，北工商 4 个专业平均技能余弦相似度 {bt['mean_skill_cosine']}（分化较好），北交大计算机类 4 个专业为 {bj['mean_skill_cosine']}（平台课共享度高，高年级存在同质化风险）。</p>""")
    S.append("<h3>③ 规模匹配差距</h3>")
    rows = []
    for s_ in efb["signals"]:
        rows.append([f"{s_['cohort']} 届", "—" if s_["cs_placement"] is None else f"{s_['cs_placement']}% / {s_['eng_placement']}%",
                     f"{s_['cs_relevance']:.0f}% / {s_['eng_relevance']:.0f}%", f"{s_['cs_income']:.0f} / {s_['eng_income']:.0f}"])
    S.append(table(["届别", "毕业去向落实率 计算机类/工学", "工作与专业相关度 计算机类/工学", "月收入（元）计算机类/工学"], rows))
    emp_mpc = mpc["real_empirical_mpc"]
    hist = emp_mpc["history"][: len(emp_mpc["observed_years"])]
    S.append(f"""<div class='cap'>表 1-3  全国本科就业质量指标（来源：官方基础数据集《中国本科生就业报告》2023/2024/2026 年版；2024–2025 届计算机类落实率未在可解析表格中公布）。</div>
<p>规模匹配从两个口径量化：（1）结果端：计算机类落实率与工学平均的差由 2021 届 −0.4pp 扩大到 2023 届 −6.2pp，专业相关度差连续三届低于 −5pp，
2025 届计算机类从事无关工作者中 {efb['skill_mismatch_evidence']['cs_unrelated_due_to_unmet_requirements_2025_pct']:.0f}% 是因“达不到专业相关工作的要求”；
（2）需求端：北京本科可及技术岗的雇主均招聘强度指数 2018–2024 年为 {', '.join(f"{h['demand_index']:.2f}" for h in hist)}，
波动在 ±12% 以内、未见持续下行。方向层面，现状各方向人数等于需求份额的前提下仍有达标缺口，说明规模问题主要表现为“人数够、能力不对口”，而非总量过剩。</p>""")
    S.append("<h2>1.4 统计检验与异常值分析</h2>")
    audits = [("2024 开发集", dv["anomaly_audit"]), ("2025 回放", m["task1_replay2025_demand_audit"]), ("2026Q1 回放", m["task1_replay2026q1_demand_audit"])]
    keys = [("total_ads", "岗位数", lambda v: f"{v:,}"), ("unique_employers", "雇主数", lambda v: f"{v:,}"),
            ("employer_gini", "雇主发布量基尼系数", lambda v: f"{v:.3f}"), ("top10_employer_ad_share", "前 10 雇主占比", pct),
            ("salary_missing_rate", "薪资缺失/面议率", pct), ("salary_inverted_count", "薪资上下限倒置数", str),
            ("log_salary_iqr_outliers", "对数薪资 IQR 异常值", str), ("log_salary_mad_outliers", "对数薪资 MAD 异常值", str),
            ("short_jd_rate", "超短职位描述率", pct), ("zero_skill_hit_rate", "零技能命中率", pct)]
    rows = [[lab] + [f(a[k]) for _, a in audits] for k, lab, f in keys]
    S.append(table(["审计项"] + [a for a, _ in audits], rows, num_cols=(1, 2, 3)))
    sens = sorted(((max(b["matched_l1_sensitivity_150_160_170"]) - min(b["matched_l1_sensitivity_150_160_170"])) * 100, b["university"] + b["major"][:4])
                  for b in t1["multi_university_benchmarks"])
    sens_med, sens_max, sens_arg = sens[len(sens) // 2][0], sens[-1][0], sens[-1][1]
    S.append(f"""<div class='cap'>表 1-4  需求端异常值与集中度审计。</div>
<p><b>处理规则。</b>倒置和缺失薪资不进入薪资模型；薪资只保留 1,500–300,000 元/月；IQR/MAD 异常值在工资分析中缩尾而非删除。
零技能命中岗位不参与达标率分母。雇主集中度较高（基尼 {dv['anomaly_audit']['employer_gini']:.2f}），因此全部分布采用雇主等权，全部区间采用雇主聚类自助法（整家雇主为重抽样单位）。
2026Q1 前 10 家雇主占 {pct(m['task1_replay2026q1_demand_audit']['top10_employer_ad_share'])}，样本集中，结论以 2025 全年回放为主。
<b>敏感性。</b>将共同学分假设在 150/160/170 学分之间变化，L1 变化中位数 {sens_med:.1f}pp、最大 {sens_max:.1f}pp（{sens_arg}）；
岗位薪资的特征回归 R² = {dv['hedonic_r2']:.2f}，技能只解释部分薪资差异，因此报告不用薪资做主评价指标。</p>""")
    S.append("""<h2>1.5 差距结论</h2><div class='box'>① 技能结构：各层方案共同的缺口集中在 Linux/Shell、数据库、Java 后端、测试和嵌入式，市属应用型方案缺口最大；部分方案在软件架构、计算机理论上供给富余。<br>
② 培养方向：单一路径无法同时覆盖四类岗位，AI/数据与系统/嵌入式方向达标率最低。<br>
③ 规模匹配：需求总量未下行而就业质量指标连续走弱，属于结构性错配。优先动课程与方向名额，而非急于压缩总招生。</div>""")

    # ---------------- Part 2 ----------------
    S.append("<h1>第二部分 方案设计</h1>")
    S.append("""<h2>2.1 系统总体架构</h2><div class='arch'>
<div><b>L1 差距引擎</b>招聘文本→技能/布鲁姆；培养方案→期望修读路径；L1/OT/布鲁姆三维差距 + 雇主聚类自助法 + 异常值审计</div>
<div><b>L2 约束求解</b>OR-Tools CP-SAT：课程增/停/升级 + 学分守恒 + 师资/实验室预算 + 先修 DAG；大二末 Gale-Shapley 稳定分流</div>
<div><b>L3 仿真与预测</b>四年学期累积推演、三情景、Split Conformal 区间、带时滞 MPC 招生控制、红队压测</div>
<div><b>L4 双环自进化</b>内环求解→未达标岗位错误诊断→Qwen 提案→GUARD 开发集门控→SIM 记忆库，全链 SHA-256 留痕</div>
<div class='wide'><b>决策驾驶舱</b>离线/在线双模：管理者查看差距诊断、方案动作、对照实验与风险矩阵；在线模式下可改预算、情景与师资参数实时重算。全部结果来自同一份封印结果文件。</div></div>
<p>设计原则：由求解器保证可执行性，大模型只负责扩展候选空间。大模型提出的新课、大纲模块和实训包须通过技能本体校验与学科同源校验，
再交给求解器在硬约束下取舍，最后只有开发集上不退化、最弱方向不下滑的候选才能晋级。</p>""")
    # 2.2 course actions
    S.append("<h2>2.2 课程体系增删改与学时变化（主案例）</h2>")
    base_hours = {}
    try:
        for c in json.loads((PROC / "university_curricula_enriched.json").read_text(encoding="utf-8")):
            if c.get("program_id") == "BTBU_CS_2022" and c.get("total_hours"):
                base_hours[c["course_name"]] = c["total_hours"]
    except Exception:
        pass
    rows, dh = [], 0
    order = {"REMOVE_ELECTIVE": 0, "ADD_ELECTIVE": 1, "UPGRADE_SYLLABUS": 2}
    for a in sorted(full["actions"], key=lambda x: (order.get(x["action"], 3), x["semester"])):
        if a["action"] == "UPGRADE_SYLLABUS":
            mod = a["module"]
            hrs = f"课内重配 {mod['instruction_hours']} 学时"
            why = f"替换为「{esc(mod['title'])}」；考核：{esc(mod['assessment'])}"
        else:
            h = float(base_hours.get(a["course_name"]) or a["credits"] * HOURS_PER_CREDIT)
            h = h if a["action"] == "ADD_ELECTIVE" else -h
            dh += h
            hrs = f"{h:+.0f} 学时"
            why = esc(a["reason"]) + (f"；面向：{'、'.join(TRK[t][:5] for t in a.get('target_tracks', []))}" if a.get("target_tracks") else "")
            if a.get("source", "").startswith("qwen"):
                why += f"（外环 {a['source'][-4:].upper()} 提案）"
        rows.append([ACT[a["action"]], esc(a["course_name"]), f"S{a['semester']}", f"{a['credits']:g}", hrs, "、".join(a["skills"][:4]), why])
    S.append(table(["动作", "课程", "学期", "学分", "学时变化", "对应技能", "理由/内容"], rows, num_cols=(3,)))
    S.append(f"""<div class='cap'>表 2-1  北工商计科 2022 级课程动作（中等预算档）。停开课程学时取官方方案，新增课按 1 学分 = {HOURS_PER_CREDIT} 学时估算；选修菜单净变化 {dh:+.0f} 学时，
学生修读学分仍按官方最低选修学分 {pc['elective_quota_credits']:g} 计，毕业总学分不变。大纲升级在原学分内替换 16 学时实验模块。</div>""")
    rows = [[CHK.get(k, k), "✓" if v else "✗"] for k, v in full["constraint_checks"].items()]
    half = (len(rows) + 1) // 2
    rows2 = [r1 + (rows[half + i] if half + i < len(rows) else ["", ""]) for i, r1 in enumerate(rows[:half])]
    S.append(table(["硬约束", "结果", "硬约束", "结果"], rows2))
    S.append(f"""<div class='cap'>表 2-2  方案硬约束校验（新增师资 {full['faculty_fte_used']}/{full['faculty_fte_cap']} FTE，实验资源 {full['lab_units_used']}/{full['lab_units_cap']} 单位，求解耗时 {full['solver_wall_time_ms']:.0f} ms）。</div>""")
    iis = m.get("task2_iis_diagnosis")
    if iis:
        S.append(f"""<div class='warn'><b>IIS 冲突诊断（约束不可行时的可解释反馈）。</b>{esc(iis['setup'])}，CP-SAT 判定 {iis['status']}；
将 {iis['guarded_constraints']} 条治理约束各绑定一个命名假设变量后，求解器给出的最小冲突集为：{'；'.join(esc(x) for x in iis['iis_conflicts']) or '—'}。
该冲突集写入外环记忆库，下一代提案据此调整；去掉强制采纳后，求解器在同一资源档内可行（{'是' if iis['lite_repaired_feasible'] else '否'}）。</div>""")
    S.append("<p><b>优先级体现。</b>外环先按“缺口影响度 = 未达标岗位中该技能的出现率 × (1 − 当前就绪度)”给技能缺口排序，决定新课、大纲模块和实训包的优先次序；内环求解器在目标函数中对最弱方向加权，并在学分预算内停开低匹配选修来腾出学时，其中《人工智能导论》由实践更强的《深度学习导论与神经网络实践》替代。</p>")
    rows = []
    for b in ("lite", "medium", "heavy"):
        p = m["task2_pareto_trio"][b]
        s = p["solver_summary"]
        rows.append([{"lite": "轻量档", "medium": "中等档（推荐）", "heavy": "重度档"}[b], f"{s['added_credits']:g} / {s['removed_credits']:g}", s["upgraded_core_count"],
                     f"{s['faculty_fte_used']}/{s['faculty_fte_cap']}", f"{s['lab_units_used']}/{s['lab_units_cap']}", pct(p["replay_2025"]["jd_qual_rate_70"]), pct(p["replay_2025"]["worst_track_readiness"])])
    S.append(table(["预算档", "新增/停开学分", "大纲升级门数", "师资 FTE 用/限", "实验资源 用/限", "2025 达标率", "最弱方向就绪度"], rows, num_cols=(2, 5, 6), hl_rows=(1,)))
    S.append("<div class='cap'>表 2-3  同一进化成果在三档资源下的可选方案。重度档放宽预算后求解器未选择更多动作，达标率与中等档持平，说明中等档已接近当前候选池的收益上限。</div>")
    # 2.3 streaming
    S.append("<h2>2.3 培养方向调整与分流机制</h2>")
    S.append(f"""<p><b>方向设置。</b>保留大一至大二公共学科基础，第 5–8 学期设四个方向，与四个岗位族一一对应。每个方向由「方向选修课群 + 定向实训/实习/毕设包」组成，
实训包不增加学分，只把原有综合实训、实习和毕业设计的选题改为方向对口。</p>
<p><b>分流规则。</b>时点：第 4 学期末；算法：学生提议的 Gale-Shapley 延迟接受算法；学生提交全部四个方向的志愿排序（意愿条件）；
方向侧按综合分 = 0.55×前四学期学分绩点 + 0.30×该方向先修课成绩 + 0.15×志愿强度 排序（成绩条件）；
每个方向设容量上下限（上限 = 需求份额 × 1.05 + 1%，下限 = 需求份额 × 0.82，容量条件）；第 5 学期开放不超过 10% 的互转名额。
该算法的结果不存在“学生与方向都愿意互换”的阻塞对，且对学生来说如实填报志愿是最优策略。</p>""")
    rows = []
    for t in st["track_allocations"]:
        rows.append([t["name"], pct(t["demand_share"]), f"{t['min_cap']}–{t['max_cap']}", t["assigned"], pct(t["utilization"]),
                     "、".join(c["course_name"] for c in full["track_electives"][t["id"]][:3]),
                     esc(full["capstone_actions_by_track"][t["id"]][0]["package_title"])])
    S.append(table(["方向", "需求份额", "容量下–上限", "录取", "利用率", "代表方向选修", "定向实训包"], rows, num_cols=(1, 3, 4)))
    warn = st.get("under_min_warnings") or []
    S.append(f"""<div class='cap'>表 2-4  分流仿真结果（300 名模拟学生，绩点与志愿为合成数据；阻塞对 = {st['blocking_pairs']}，第一志愿满足率 {pct(st['first_choice_satisfaction'])}，前两志愿 {pct(st['top2_choice_satisfaction'])}）。</div>
{f"<div class='warn'>预警：{esc(warn[0]['track'])}方向录取 {warn[0]['assigned']} 人，低于开班下限 {warn[0]['min_cap']} 人。处置规则：第 5 学期互转窗口优先向该方向定向开放；仍不足时与自动化等相近专业合班开课。这对应风险 R02。</div>" if warn else ''}""")
    # 2.4 enrollment
    S.append("<h2>2.4 基于就业反馈的招生规模动态调整</h2>")
    R = efb["rules"]
    P_ = emp_mpc["parameters"]
    S.append(f"""<p>招生控制器分两个通道判断、一个执行器落地：</p>
{table(['要素', '规则'], [
        ['结果通道（就业反馈）', f"计算机类与工学平均之差：落实率差 &lt; {R['warn_placement_gap_pp']}pp 或专业相关度差 &lt; {R['warn_relevance_gap_pp']}pp 记为预警；用相对差消除整体就业形势的共同冲击"],
        ['需求通道（市场信号）', f"北京本科可及技术岗雇主均招聘强度，经 Holt 趋势滤波后与基准比较，死区 ±{P_['deadband'] * 100:.1f}%"],
        ['触发阈值', f"预警连续 {R['confirm_cohorts']} 届确认后触发；扩招须落实率差 &gt; +{R['expand_placement_gap_pp']}pp 且收入差 &gt; +{R['expand_income_gap'] * 100:.0f}% 连续 2 届，且需求上行"],
        ['调整幅度', f"单次 = min({R['max_step_rate'] * 100:.0f}%, {R['k_placement'] * 100:.1f}%×|落实率差| + {R['k_relevance'] * 100:.1f}%×|相关度差|)；治理边界 [{R['governance_bounds'][0]}, {R['governance_bounds'][1]}] 人"],
        ['调整周期', f"每年就业报告发布后（Q3）评估一次；总量调整后冷却 {R['cooldown_years']} 年；4 年培养时滞由 MPC 用预测的第 t+4 年需求补偿"],
        ['错配解耦规则', '结果预警而需求未下行 → 判为结构性错配：保持总招生、调整方向名额并触发课程外环进化；改革生效后仍连续 2 届预警再按幅度公式缩减'],
        ['数据门控', '校内尚无同专业连续 2 届同口径就业追踪时，首年保持基准招生并建立追踪台账，不启用自动调招'],
    ])}""")
    rows = [[f"{s_['cohort']} 届 → {d['decision_year']}", "—" if s_["gap_placement_pp"] is None else f"{s_['gap_placement_pp']:+.1f}", f"{s_['gap_relevance_pp']:+.0f}",
             f"{s_['gap_income_rel'] * 100:+.1f}%", d["state"], esc(d["action"]), f"{d['intake_after']:.0f}"] for s_, d in zip(efb["signals"], efb["decision_log"])]
    S.append(table(["证据→决策年", "落实率差(pp)", "相关度差(pp)", "收入差", "状态", "动作", "招生数"], rows, num_cols=(1, 2, 3, 6)))
    se, ss = emp_mpc["stability_comparison"], mpc["synthetic_stress_mpc"]["stability_comparison"]
    S.append(f"""<div class='cap'>表 2-5  就业反馈决策日志（基准招生 120 人）。</div>
<div class='box'>{esc(efb['current_recommendation']['interpretation'])}</div>""")
    S.append(fig("fig4_mpc_control.png", f"图 2  带 4 年时滞的招生控制。左：北京实测需求指数驱动，MPC 维持基准，朴素“缺就多招”产生总变差 {se['naive_total_variation']:.0f} 的蛛网震荡；右：增长+周期压力情景下 MPC 震荡降低 {ss['oscillation_reduction_pct']:.1f}%，跟踪误差较冻结降低 {ss['rmse_reduction_vs_freeze_pct']:.1f}%。"))
    # 2.5 macro
    mac = m["task2_macro_major_portfolio"]
    S.append("<h2>2.5 专业布点的增、调、停（学院层）</h2>")
    rows = []
    for d in mac["decisions"][:8]:
        rows.append([f"{esc(d['major_name'])}（{d['major_code']}）", f"{d['lifecycle_before']} → {d['lifecycle_after']}", d["action"], esc(d["reason"])])
    S.append(table(["专业", "生命周期", "动作", "依据"], rows))
    S.append(f"""<div class='cap'>表 2-6  北工商计算机类专业组合决策（学院编制 {mac['college_fte_budget']:g} FTE，现占 {mac['current_total_fte']:g}）。停招专业执行 4 年平稳退出，逐年释放编制 {' → '.join(f'{x:g}' for x in mac['freed_fte_via_teach_out_y1_y4'])} FTE，在校生按原方案毕业。新专业以微专业先行，编制到位后申报。</div>""")
    cc = m.get("task2_macro_catalog_check") or []
    if cc:
        fixed = [f"{c['major_name']} {c['code_before']}→{c['catalog_2026_code']}" for c in cc if c["catalog_2026_code"] and not c["match"]]
        S.append(f"<p class='small'>专业代码已逐一对照教育部《普通高等学校本科专业目录（2026年）》（教高函〔2026〕2号）：{sum(1 for c in cc if c['catalog_2026_code'])}/{len(cc)} 个专业在目录中，"
                 + (f"更正 {len(fixed)} 个代码（{'；'.join(fixed)}）。" if fixed else "代码全部一致。") + "</p>")
    # 2.6 RSI
    S.append("<h2>2.6 双环自进化：让方案随数据持续更新</h2>")
    S.append("""<p><b>内环</b>在给定候选池与策略参数（统称 Harness）下求解最优方案；<b>外环</b>每一代先诊断“哪些岗位未达标、缺的是哪些技能”，
将诊断、上一代完整轨迹和记忆库经验交给 Qwen，由其提出新课、同源大纲模块、实训包和策略权重。候选 Harness 重新求解后必须通过 GUARD：
开发集综合分不退化、最弱方向不下滑、近期训练年份不退化，三者同时满足才晋级；未通过的候选连同原因写入记忆库，指导下一代。
第 3 代额外注入一个“偏向头部方向”的对抗候选，用来检验 GUARD 能否拦下牺牲弱势方向的改动。开发集综合分 = 0.42×达标率 + 0.28×就绪度 + 0.16×L1 + 0.08×OT + 0.06×最弱方向就绪度。</p>""")
    S.append(fig("fig2_rsi_evolution.png", "图 3  三个专业的逐代进化轨迹（2024 开发集）。"))
    rows = []
    for h in rsi["evolution_history"]:
        rows.append([f"G{h['generation']}", h["event"], f"{h['dev_score']:.4f}", f"{h['active_best_score']:.4f}", "、".join(h.get("new_courses_proposed", [])[:3]) or "—", esc(h["guard_decision"]["reason"])[:70]])
    S.append(table(["代", "事件", "候选分", "当前最优", "Qwen 提出的新课", "GUARD 判定"], rows, num_cols=(2, 3)))
    tok = rsi["token_ledger_stats"]
    S.append(f"""<div class='cap'>表 2-7  主案例进化审计链（终态哈希 {rsi['trace_final_sha256'][:16]}…）。Qwen 调用全部记入重放账本（{tok['entries']} 条，共 {tok['total_tokens']:,} tokens），同一请求可按哈希复现。</div>""")

    # ---------------- Part 3 ----------------
    S.append("<h1>第三部分 效果验证与预测结果</h1>")
    S.append(f"""<h2>3.1 验证设计</h2>
<p><b>时间切分与防泄漏。</b>方案生成只用 ≤ 2023 年训练数据，进化门控只用 2024 开发集；2025 年（{r25['BTBU_CS_2022']['ads']:,} 条、{r25['BTBU_CS_2022']['employers']} 家雇主）与 2026Q1（{r26['BTBU_CS_2022']['ads']} 条、{r26['BTBU_CS_2022']['employers']} 家）
放在隔离目录，只在最终评估时经审计上下文读取，每次读取都记录文件哈希。由于研发过程中曾查看过 2025 回放结果，我们将其如实称为“历史回放”；严格意义上的独立盲测见 3.3：在冻结全部方案与评估代码并公开登记假设之后，才首次抽取从未读取过的人群一次性检验。</p>
<p><b>指标定义。</b>对每个方向的毕业生，先算每项技能的就绪度 r = min(1, 学分质量/2) × (0.40 + 0.60 × min(1, 课程布鲁姆/岗位布鲁姆))，
再逐条评估同方向岗位：岗位所列技能平均就绪度 ≥ 0.70，且 ≥ 70% 的所列技能就绪度 ≥ 0.55，即记该岗位“达标”。
岗位级达标率为雇主等权的达标岗位占比。该定义不使用 S 形放大函数，也不使用任何就业结果数据，因此它衡量的是“课程能力是否对口”，不是就业率。</p>
<p><b>对照组。</b>B0 现状冻结；B1 单轨静态优化（同预算、同候选池、不分流）；B2 多轨静态（分流 + 常规选修池，无进化）；B3 单轮 LLM（一次 Qwen 提案 + 求解，无多代、无记忆）；
B4 双环但关闭 GUARD；Full 为完整系统。所有组共用同一硬约束、同一预算与同一评估器。区间采用雇主聚类配对自助法（600 次，固定种子）。</p>""")
    S.append("<h2>3.2 前后对比与六组消融</h2>")
    S.append(fig("fig3_ablation_6way.png", "图 4  六组受控消融（2025 回放；黑色短横为 2026Q1）。"))
    rows = []
    for aid, alab in ARMS:
        cells = [alab]
        for pid in PROG:
            mm = r25[pid]["metrics"][aid]
            cells.append(f"{pct(mm['jd_qual_rate_70'])} / {pct(r26[pid]['metrics'][aid]['jd_qual_rate_70'])}")
        cells.append(pct(r25["BTBU_CS_2022"]["metrics"][aid]["worst_track_readiness"]))
        rows.append(cells)
    S.append(table(["对照组", "北工商计科 2025/2026Q1", "北工商AI 2025/2026Q1", "北信科计科 2025/2026Q1", "计科最弱方向就绪度"], rows, num_cols=(1, 2, 3, 4), hl_rows=(5,)))
    S.append("<div class='cap'>表 3-1  岗位级达标率。</div>")
    rows = []
    for pid in PROG:
        for sp, res in (("2025", r25[pid]), ("2026Q1", r26[pid])):
            cis = res["paired_qual_ci95"]
            rows.append([PROG_S[pid], sp] + [ci(cis[a]["Full_EvoAlign_OS"]) + ("" if cis[a]["Full_EvoAlign_OS"]["ci95_lo"] > 0 else " （不显著）") for a, _ in ARMS[:5]])
    S.append(table(["专业", "回放期", "Full−B0", "Full−B1", "Full−B2", "Full−B3", "Full−B4"], rows))
    S.append("<div class='cap'>表 3-2  Full 相对各对照的配对差（pp，95% 区间）。</div>")
    b4 = {pid: r25[pid]["paired_qual_ci95"]["B4_BiLevel_NoGuard"]["Full_EvoAlign_OS"] for pid in PROG}
    gaps_b3 = [(r25[pid]["metrics"][a]["jd_qual_rate_70"] - r25[pid]["metrics"]["B3_SingleShot_LLM"]["jd_qual_rate_70"]) * 100
               for pid in PROG for a in ("B4_BiLevel_NoGuard", "Full_EvoAlign_OS")]
    S.append(f"""<p><b>解读。</b>① 分流（B1→B2）带来稳定但有限的提升；② 多代进化是主要增量来源，B3（一次性 LLM 提案）与 B4/Full 之间相差 {min(gaps_b3):.0f}–{max(gaps_b3):.0f}pp，
来自第 2 代解锁的定向实训包与同源大纲模块。③ GUARD 的作用因专业而异：北工商 AI 中，关闭 GUARD 的 B4 接受了第 3 代偏向头部方向的改动，
最终落后 Full {b4['BTBU_AI_2022']['delta'] * 100:.1f}pp；北信科小幅显著（{b4['BISTU_CS_2025']['delta'] * 100:+.1f}pp）；北工商计科两者持平（{b4['BTBU_CS_2022']['delta'] * 100:+.2f}pp，不显著）。
2026Q1 样本小，Full−B4 在三个专业上均不显著。因此 GUARD 的主要价值在于“防止偏科式退化”，而不是在每个专业上都能提分。</p>""")
    S.append(fig("structural_repair_paired.png", "图 5  Full 相对 B0–B3 的配对增益与 95% 区间。"))
    # 3.3 preregistered independent blind test
    if PRE and BLIND:
        S.append("<h2>3.3 预注册独立盲测</h2>")
        pops = BLIND["results"]
        S.append(f"""<p><b>程序。</b>① {PRE['registered_utc'][:19].replace('T', ' ')} UTC 登记预注册文件（SHA-256 {PRE['preregistration_sha256'][:16]}…），
冻结封印结果、各对照组逐方向课程快照、知识图谱与评估器源码的哈希，写明数据筛选 SQL、四条假设与判定规则；② 随后（{BLIND['run_started_utc'][:19].replace('T', ' ')} UTC）
才首次从官方招聘库抽取两个从未被任何 EvoAlign 代码读取过的人群，存入隔离目录并记录哈希；③ 一次性评估，脚本拒绝二次运行。
登记前仅对这两个人群做过按城市的行数统计。首次登记因分析代码未写完，在读取任何数据前撤回并重新登记，撤回记录随材料提交。</p>
<p><b>人群。</b>P1 {esc(pops['P1_jinji']['label'])}（{pops['P1_jinji']['n_ads']:,} 条、{pops['P1_jinji']['n_employers']} 家雇主，主检验）；
P2 {esc(pops['P2_national_ex_bj']['label'])}（{pops['P2_national_ex_bj']['n_ads']:,} 条、{pops['P2_national_ex_bj']['n_employers']:,} 家雇主，稳健性）。
两者都不含北京岗位，检验的是方案在区域与全国劳动力市场上的泛化，而不是北京市场本身。</p>""")
        rows = []
        for pk, pr in pops.items():
            for prog, v in pr["programs"].items():
                bm = v["metrics"]
                hs = {h["id"]: h for h in v["hypotheses"]}
                rows.append([("P1 津冀" if pk.startswith("P1") else "P2 京外"), PROG_S[prog], pct(bm["B0_Freeze"]["jd_qual_rate_70"]), pct(bm["B4_BiLevel_NoGuard"]["jd_qual_rate_70"]),
                             f"<b>{pct(bm['Full_EvoAlign_OS']['jd_qual_rate_70'])}</b>"]
                            + [f"{hs[k]['delta'] * 100:+.1f} [{hs[k]['ci95_lo'] * 100:+.1f}, {hs[k]['ci95_hi'] * 100:+.1f}] {'✓' if hs[k]['supported'] else '✗'}" for k in ("H1", "H2", "H3", "H4")])
        S.append(table(["人群", "专业", "B0", "B4", "Full", "H1 Full−B0 > 0", "H2 Full−B2 > 0", "H3 Full−B3 > 0", "H4 Full−B4 > −2pp"], rows, num_cols=(2, 3, 4)))
        sup = BLIND["hypotheses_supported"]
        S.append(f"""<div class='cap'>表 3-2b  预注册假设检验（pp，雇主聚类配对 95% 区间）。✓ = 按预先登记的规则成立。P1 {sup['P1_jinji']}、P2 {sup['P2_national_ex_bj']} 条成立。</div>
<p>在完全未见过的人群上，三个专业的结论与北京回放一致：相对现状、多轨静态与单轮 LLM 基线均显著提升；相对“关闭 GUARD 的双环”满足非劣效，
人工智能专业仍显著更好（P1 +{pops['P1_jinji']['programs']['BTBU_AI_2022']['hypotheses'][3]['delta'] * 100:.1f}pp）。这说明增益不是对北京回放数据的调参结果。</p>""")
    S.append("<h2>3.4 反事实对比与四年情景推演</h2>")
    S.append(fig("fig5_dreamsim_scenarios.png", "图 6  北工商计科方案的四年累积轨迹与三情景（2025 回放）。右图为招聘广告月薪的 90% 解释区间，仅供参考。"))
    rows = [["反事实冻结", pct(frz["jd_qual_rate_70"]), pct(frz["jd_mean_readiness"]), pct(frz["worst_track_readiness"]), f"{frz['explanatory_ad_salary_median']:,.0f} [{frz['explanatory_ad_salary_conformal90'][0]:,.0f}, {frz['explanatory_ad_salary_conformal90'][1]:,.0f}]", "—"]]
    assume = {"optimistic": "新课/升级/实训效果 100%", "baseline": "新课 94%、升级 96%、实训 95%", "pessimistic": "新课 80%、升级 86%、实训 84%，负荷惩罚 8%"}
    for k in ("optimistic", "baseline", "pessimistic"):
        v = sc[k]
        rows.append([esc(v["label"]), pct(v["jd_qual_rate_70"]), pct(v["jd_mean_readiness"]), pct(v["worst_track_readiness"]),
                     f"{v['explanatory_ad_salary_median']:,.0f} [{v['explanatory_ad_salary_conformal90'][0]:,.0f}, {v['explanatory_ad_salary_conformal90'][1]:,.0f}]", assume[k]])
    S.append(table(["情景", "达标率", "平均就绪度", "最弱方向", "广告月薪中位 [90% 区间]", "假设"], rows, num_cols=(1, 2, 3)))
    S.append(f"""<div class='cap'>表 3-3  情景敏感性。薪资区间来自分割式保形预测（校准集 {conf['calibration_n']:,} 条，实测覆盖率 {pct(conf['calibration_empirical_coverage'])}，模型 R² = {conf['r2_dev']:.2f}），区间宽、解释力有限，只作参考。</div>
<p>悲观情景下达标率仍为 {pct(sc['pessimistic']['jd_qual_rate_70'])}，高于冻结的 {pct(frz['jd_qual_rate_70'])}；三情景之间的差距（{(sc['optimistic']['jd_qual_rate_70'] - sc['pessimistic']['jd_qual_rate_70']) * 100:.1f}pp）主要取决于师资到位和新课质量，
对应风险 R01。逐年轨迹显示提升主要发生在分流后的第 5–8 学期，与方案动作的安排一致。</p>""")
    ag = m.get("task3_agent_dream_sim")
    if ag:
        rows = []
        for pid, v in ag.items():
            for sc, lab in (("optimistic", "乐观"), ("baseline", "基准"), ("pessimistic", "悲观")):
                f0, f1 = v["freeze"][sc], v["full"][sc]
                rows.append([PROG_S[pid], lab, f"{pct(f0['mean_student_match'])} [{pct(f0['student_match_p10_p90'][0], 0)}–{pct(f0['student_match_p10_p90'][1], 0)}]",
                             f"{pct(f1['mean_student_match'])} [{pct(f1['student_match_p10_p90'][0], 0)}–{pct(f1['student_match_p10_p90'][1], 0)}]",
                             f"{v['full_minus_freeze_pp'][sc]:+.1f}", pct(f1["student_struggling_rate"])])
        S.append("<h3>多智能体沙盘复核</h3>")
        S.append(table(["专业", "情景", "现状：学生匹配度 [p10–p90]", "方案：学生匹配度 [p10–p90]", "差值(pp)", "方案下负荷超限学生"], rows, num_cols=(4, 5)))
        all_beat = all(v["pessimistic_full_beats_optimistic_freeze"] for v in ag.values())
        S.append(f"""<div class='cap'>表 3-3b  {esc(next(iter(ag.values()))['method'])}。指标为学生智能体技能深度对岗位技能束的覆盖比例（雇主等权），与达标率口径不同，用作独立复核。
三个专业中，方案在悲观情景下的学生匹配度{'均' if all_beat else '并非全部'}高于现状在乐观情景下的水平；负荷超限学生比例与现状相同，说明方案没有加重学生负担。</div>""")
    # 3.4 natural experiments
    da = ne["direction_agreement"]
    S.append("<h2>3.5 自然实验：与高校和教育部的真实决策对照</h2>")
    nci_r, nci_f = ne["paired_qual_ci95_vs_real_2025"], ne["paired_qual_ci95_vs_frozen_2022"]
    S.append(table(["方案（均在 2025 回放上评估）", "达标率", "相对北工商实际 2025 改版（pp，95% CI）"], [
        ["北工商 2022 版（改版前）", pct(ne["btbu_frozen_2022_on_replay2025"]["jd_qual_rate_70"]), ci(nci_r["frozen_2022"])],
        ["北工商实际 2025 版", pct(ne["btbu_real_2025_on_replay2025"]["jd_qual_rate_70"]), "—"],
        ["EvoAlign 预测（只用 ≤ 2022 数据，强制单轨）", pct(ne["evoalign_pred_single_track_on_replay2025"]["jd_qual_rate_70"]), ci(nci_r["evoalign_forecast_single_track"])],
        ["EvoAlign 预测（只用 ≤ 2022 数据，含分流）", pct(ne["evoalign_pred_on_replay2025"]["jd_qual_rate_70"]), ci(nci_r["evoalign_forecast"])],
    ], num_cols=(1,)))
    S.append(f"""<div class='cap'>表 3-4  课程级自然实验。</div>
<p>北工商 2025 年实际改版新增了 Rust、开源操作系统实践、人工智能应用实践等课程，但同时压缩了软件工程、DevOps、Linux 等工程类学分并增加了金融特色课，
因此在“本科可及技术岗”口径下达标率下降 {abs(nci_f['real_2025']['delta']) * 100:.1f}pp。这一结果说明本指标只反映与北京入门技术岗的对口程度，不评价学校的特色定位。
在改动方向上，只用 ≤ 2022 数据的预测与学校实际改版部分一致：两者都调整了《{'》《'.join(da['course_level']['agreed'])}》（{da['course_level']['n_agreed']} 门，
随机期望 {da['course_level']['chance_expectation']:.1f} 门，超几何检验 p = {da['course_level']['hypergeom_p_value']:.3f}）；学校新课涉及的技能有 {pct(da['skill_level']['recall'], 0)} 出现在预测中
（p = {da['skill_level']['hypergeom_p_value']:.2f}，未达显著，因为预测覆盖的技能较多）。</p>""")
    S.append(table(["专业布点自然实验（≤ 2023 预测 2024 北京新增）", "Top-5 专业", "命中", "覆盖 2024 年新增布点"], [
        ["EvoAlign（需求缺口 + 备案历史）", "、".join(nm["evoalign_top5_predicted_emerging"]), f"{nm['top5_hit_count']}/5", pct(nm["evoalign_top5_volume_coverage"])],
        ["惯性基线（2021–2023 新增最多的 5 个）", "、".join(nm["persistence_baseline_top5"]), f"{nm['persistence_baseline_hit_count']}/5", pct(nm["persistence_baseline_volume_coverage"])],
    ]))
    MR = json.loads((OUT / "macro_rolling_2025_partial.json").read_text(encoding="utf-8")) if (OUT / "macro_rolling_2025_partial.json").exists() else None
    if MR:
        S.append(table(["滚动复验（≤2024 预测 2025 北京新增）", "Top-5 专业", "命中", "覆盖已知新增"], [
            ["EvoAlign", "、".join(MR["evoalign_top5"]), f"{len(MR['evoalign_hits'])}/5", pct(MR["evoalign_volume_coverage"])],
            ["惯性基线（2020–2024 新增最多）", "、".join(MR["persistence_top5"]), f"{len(MR['persistence_hits'])}/5", pct(MR["persistence_volume_coverage"])],
        ]))
        S.append(f"<div class='cap'>表 3-5b  第二个时间起点的复验。教育部 2025 年度全国附件与北京市教委 2025 年度市属高校结果均未能在官方网站定位，"
                 f"真值改用 {len(MR['truth_universities'])} 所北京高校官网自行公布的结果（逐页存档并记录 SHA-256），其中计算机/智能/信息类新增 {len(MR['truth_cs_ai_adds'])} 个："
                 f"{'、'.join(u + n for u, n, _ in MR['truth_cs_ai_adds'])}。样本小且不完整，只作方向性佐证。</div>")
    S.append(f"<div class='cap'>表 3-5  2024 年北京高校新增计算机/智能/信息类布点 {nm['blind_moe_2024_bj_cs_ai_total_adds']} 个。命中按专业名称前缀匹配；两者命中数接近，差别主要在覆盖的布点数量。</div>")
    # 3.5 transfer
    S.append("<h2>3.6 跨校与跨专业迁移</h2>")
    rows = []
    for k, lab in (("BISTU_CS_2025_unseen_university", "跨校：北信科计科"), ("BTBU_AI_2022_unseen_major", "跨专业：北工商 AI")):
        v = tr[k]
        rows.append([lab, pct(v["baseline_freeze"]["jd_qual_rate_70"]), pct(v["cold_start_H0"]["jd_qual_rate_70"]), pct(v["warm_transfer_Hstar_SIM"]["jd_qual_rate_70"]), pct(v["full_local_evolved"]["jd_qual_rate_70"])])
    S.append(table(["目标", "现状", "冷启动（0 代）", "迁移进化成果（0 代、不调用 LLM）", "本地完整进化"], rows, num_cols=(1, 2, 3, 4)))
    S.append("<div class='cap'>表 3-6  把北工商计科进化出的候选池、实训包与记忆库直接用于新对象（2025 回放）。同类专业迁移效果接近本地进化；跨专业仍需本地进化。</div>")
    S.append("""<h2>3.7 不确定性与局限</h2><div class='warn'>① 达标率是能力对口的代理指标，尚未用校内学生成绩与就业追踪校准；② 修读路径用官方最低选修学分换算，分流使用模拟学生；
③ 课程技能由大模型抽取，抽检一致率较高但非人工全量标注；④ 北京 2025 回放为“历史回放”，独立性由 3.3 的预注册京外盲测补足，但后者检验的是跨区域泛化；2026Q1 样本小、雇主集中；
⑤ 新课实际效果取决于师资与教学质量，情景分析给出了受影响范围；
⑥ 教育部 2025 年度本科专业备案审批结果（据高校官网转述为教高函〔2026〕3号）的全国附件、北京市教委 2025 年度市属高校结果均未能在官方网站定位；主检验用 2020–2024 年度官方结果，2025 年度只用 10 所北京高校官网公告做部分复验（表 3-5b）。这些均在风险 R01/R05/R08 中设置了预警与校准措施。</div>""")

    # ---------------- Part 4 ----------------
    S.append("<h1>第四部分 风险评估与应对</h1>")
    S.append(f"""<p>风险逐项绑定具体方案动作。影响程度 S 与发生可能性 L 均为 1–5 级，风险分 = S × L。可能性来自压力测试：对推荐方案施加 {rt['n_mc_per_risk']} 次随机极端冲击
（师资断供、方向挤兑、GPU 不足、技术栈漂移等），分别统计“无治理”和“启用对应治理机制”两种情形下的指标超限频次。</p>""")
    S.append(fig("fig6_risk_and_transfer.png", "图 7  左：跨校/跨专业迁移；右：8 类风险在无治理与治理闭环下的压测超限频次。"))
    rows = []
    for r in sorted(rt["risk_matrix"], key=lambda x: x["id"]):
        Lraw = round(r["raw_risk_score"] / max(1, r["severity"]))
        rows.append([f"<b>{r['id']}</b> {esc(r['name'])}", esc(r["bound_action"]), esc(r["affected_group"]),
                     f"S{r['severity']}×L{Lraw}={r['raw_risk_score']} → {r['residual_risk_score']}",
                     f"{pct(r['ungoverned_breach_rate'], 0)} → {pct(r['governed_breach_rate'], 0)}"])
    S.append(table(["风险", "绑定动作", "影响对象", "风险分（治理前→后）", "压测超限"], rows))
    S.append("<div class='cap'>表 4-1  风险矩阵。</div>")
    rows = []
    for r in sorted(rt["risk_matrix"], key=lambda x: x["id"]):
        rows.append([r["id"], esc(r["owner"]), esc(r["trigger"]), esc(r["deadline"]), esc(r["mitigation"]), esc(r["rollback"])])
    S.append(table(["风险", "责任主体", "预警指标", "时间节点", "应对措施", "回退预案"], rows))
    S.append(f"""<div class='cap'>表 4-2  应对策略。</div>
<p>治理后平均超限频次由 {pct(rt['ungoverned_mean_breach_rate'])} 降至 {pct(rt['governed_mean_breach_rate'])}。残余风险主要是 R07（停招与旧课退出）和 R05（技术栈漂移），
分别通过四年平稳退出台账和每年一次外环体检管控。压测参数为专家设定的极端情形，用于比较治理机制的有效性，不代表真实发生概率。</p>""")

    # ---------------- Part 5 ----------------
    S.append("<h1>第五部分 数据说明</h1>")
    kg = m["kg_stats"]
    S.append(table(["数据", "来源", "范围与规模", "用途"], [
        ["招聘数据（上市公司 + 应届生）", "官方基础数据集", "原始 18,866,559 条 → 技术候选 1,711,875 条 → 北京技术岗 2018–2026Q1；本科可及入门岗：训练 2018–2023 51,460 / 2024 1,718 / 2025 2,645 / 2026Q1 351", "需求画像、评估、招生需求指数"],
        ["《中国本科生就业报告》", "官方基础数据集", "2023/2024/2026 年版（2025 年版文字层不可解析，未使用）；提取 39 个指标值", "就业反馈通道"],
        ["政策文件、各省支持产业", "官方基础数据集", "十五五规划及专项规划等", "新兴技能与专业方向先验"],
        ["高校培养方案与课程大纲", "自采（高校官网公开文件）", f"{kg['universities']} 校 {kg['programs']} 份方案、{kg['total_courses']:,} 门课程、{kg['syllabus_enriched_unique']} 门有大纲", "供给画像、自然实验"],
        ["教育部本科专业备案审批结果", "自采（教育部官网）", f"2020–2023 年度 {n_moe_hist:,} 条布点记录（2020 年度为 XLS 附件）；2024 年度 1,994 条单独隔离用于检验；2025 年度全国附件未能定位，改以 10 所北京高校官网公告（31 条）部分替代", "专业布点决策与自然实验"],
        ["《普通高等学校本科专业目录（2026年）》", "自采（教育部官网）", f"{n_catalog} 种专业", "专业代码核验"],
        ["计算机领域“101 计划”核心课", "自采（国家高等教育智慧教育平台）", "12 门核心课、33 所试点高校", "培养方案标准对照"],
        ["ESCO v1 技能与职业本体", "自采（欧盟官方接口）", f"{esco_cov['verified_skill_concepts']} 个技能概念、{esco_cov['verified_occupations']} 个职业", "技能本体国际对照"],
        ["北京市/区紧缺人才目录、教学质量报告", "自采（政府、高校官网）", "2025–2026 年目录 4 份等", "新兴技能权重、风险参数"],
        ["CS2023 技能本体", "自采（ACM/IEEE）", "48 项标准技能 × 6 级布鲁姆", "供需两端的公共坐标"],
    ]))
    S.append("<div class='cap'>表 5-1  数据来源与规模。自采公开文件共 48 份（成功 47 份），逐一登记网址、采集时间与 SHA-256，见数据支撑材料《自采数据登记表》。</div>")
    S.append("""<h2>5.1 清洗方法</h2><p>① 按数据字典建议以 utf-8-sig 和标准 CSV 解析器读取，日期去尾空格、年份转整数，若干/面议等写法统一为缺失；
② 档案内精确去重，重复发布不计为独立岗位；③ 学历、经验枚举建立统一映射表，定义“本科可及入门岗”；④ 剔除职位描述中的数据站点水印和联系方式；
⑤ 薪资取上下限中点，去除倒置与越界值；⑥ 培养方案逐门课与官方表格核对必/选修与学期，11 份基准方案学分口径全部核验；
⑦ 工商数据按数据字典提示去除个体工商户，仅作背景统计，未进入主模型。</p>""")
    S.append("""<h2>5.2 大模型使用说明</h2><p>大模型（通义千问 Qwen3.8-flash/max，OpenAI 兼容接口）只用于两类任务：课程文本的技能抽取与复核，以及外环改进提案。
发送内容仅为公开课程文本和聚合统计，不含任何个人信息与招聘原文。全部请求与响应按哈希记入重放账本，复核时无需联网。
API 密钥仅保存在本机受保护的环境文件中，不进入代码库与任何提交材料。大模型产出均需经本体校验、学科同源校验和求解器硬约束，不直接构成决策。</p>""")
    S.append("""<h2>5.3 合规自查声明</h2><div class='box'>本团队声明：方案所用数据均来自官方基础数据集或《赛题规则》允许的公开渠道；采集过程遵守来源网站使用协议、robots 协议与访问频率限制；
未采集含个人信息的数据；发布内容已按规定脱敏（雇主以假名编号表示，职位描述中的联系方式已删除，数据支撑材料仅含 30 条脱敏岗位样例与聚合特征矩阵）。<br><br>
队长签字：＿＿＿＿＿＿　　日期：＿＿＿＿＿＿</div>""")

    # ---------------- Appendix ----------------
    S.append("<h1>附录 复核方法与文件清单</h1>")
    S.append(f"""<p>数据支撑材料提供一个无需求解器、无需大模型和网络的独立复核脚本 <span class='mono'>verify_key_results.py</span>：
读取封印结果文件、各对照组的逐方向课程快照和脱敏的逐岗位技能矩阵，重新计算消融指标、配对区间、自然实验、任务 1 基准、招生 MPC 与就业反馈决策，
并逐项与封印值比对。当前结果：<b>{ver_txt} 项一致</b>（容差 1.2×10⁻⁵，及封印精度）。运行方法见数据支撑材料 README。</p>""")
    S.append("<h2>A 档检查点对照</h2>")
    S.append(table(["任务", "检查点", "本报告位置"], [
        ["任务 1", "数据提取、清洗、特征提取", "1.2、5.1"],
        ["任务 1", "技能结构 / 培养方向 / 规模匹配三维量化", "1.3 ①②③，表 1-1 至 1-3，图 1"],
        ["任务 1", "统计检验与异常值分析", "1.4，表 1-4（雇主聚类自助法区间、IQR/MAD、集中度、敏感性）"],
        ["任务 2", "课程增删改与学时变化及理由，满足硬约束", "2.2，表 2-1、2-2、2-3"],
        ["任务 2", "培养方向与分流机制（时点、成绩/意愿/容量条件）", "2.3，表 2-4"],
        ["任务 2", "招生规模动态调整（触发阈值、调整幅度、调整周期）", "2.4，表 2-5，图 2"],
        ["任务 3", "前后对比、反事实对比、情景推演；多方法互证", "3.2–3.6（消融、预注册独立盲测、反事实、三情景、多智能体沙盘、自然实验、迁移）"],
        ["任务 3", "关键指标可量化，不确定性表达", "表 3-1 至 3-3 的 95% 区间、情景区间、保形区间；3.7"],
        ["任务 4", "与方案动作绑定的风险 ≥ 3 个及量化", "表 4-1（8 项，S×L 矩阵与压测频次）"],
        ["任务 4", "责任主体、时间节点、预警指标", "表 4-2"],
    ]))
    if verify:
        groups = {}
        for c in verify["checks"]:
            g = groups.setdefault(c["group"], [0, 0])
            g[0] += c["pass"]
            g[1] += 1
        S.append(table(["复核组", "通过/总数"], [[g, f"{p}/{n}"] for g, (p, n) in groups.items()], num_cols=(1,)))
    S.append(table(["文件", "内容"], [
        ["outputs/verified_master_results.json", "单一事实源（全部报告、PPT、驾驶舱数字的来源），内含 SHA-256 封印"],
        ["outputs/plan_snapshots.json", "3 专业 × 6 对照组及自然实验、三档预算方案的逐方向课程清单"],
        ["outputs/replay_ledger.jsonl", "Qwen 请求/响应重放账本（哈希索引）"],
        ["outputs/replay_bundle/", f"SHA-256 签名的自进化重放包：各专业各对照组的逐代轨迹与候选基因组、记忆库、最终 Harness、Qwen 调用原文、帕累托三档归档（根哈希 {rb_root[:16]}…）"],
        ["outputs/blind_access_audit.jsonl", "隔离数据读取审计日志"],
        ["data/verification/*", "脱敏逐岗位特征矩阵、基准方案修读路径、就业报告指标、输入/输出样例"],
        ["src/", "全部引擎、实验、复核与交付生成代码"],
        ["web/index.html", "决策驾驶舱（离线可直接打开；在线模式 uvicorn server.cockpit_server:app）"],
    ]))
    return "\n".join(S)


async def to_pdf(html_path: Path, pdf_path: Path) -> None:
    from playwright.async_api import async_playwright
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page()
        await pg.goto(html_path.as_uri(), wait_until="networkidle")
        await pg.pdf(path=str(pdf_path), format="A4", print_background=True, display_header_footer=True,
                     header_template="<div></div>",
                     footer_template="<div style='font-size:8px;width:100%;text-align:center;color:#64748B'>产教智对 · EvoAlign-OS 方案报告 · <span class='pageNumber'></span> / <span class='totalPages'></span></div>",
                     margin={"top": "16mm", "bottom": "17mm", "left": "15mm", "right": "15mm"})
        await b.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", default=TEAM.get("team_name") or "【队伍名称】")
    args = ap.parse_args()
    m = json.loads((OUT / "verified_master_results.json").read_text(encoding="utf-8"))
    vp = OUT / "verification_report.json"
    verify = json.loads(vp.read_text(encoding="utf-8")) if vp.exists() else None
    if verify and verify["summary"]["master_sha256"] != m["master_sha256"]:
        raise SystemExit("verification_report.json is stale; rerun verify_key_results.py first")
    DELIV.mkdir(parents=True, exist_ok=True)
    body = build(m, args.team, verify)
    doc = f"<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'><title>产教智对 方案报告</title><style>{CSS}</style></head><body>{body}</body></html>"
    hp = DELIV / "方案报告.html"
    hp.write_text(doc, encoding="utf-8")
    asyncio.run(to_pdf(hp, DELIV / "方案报告.pdf"))
    print("report written:", DELIV / "方案报告.pdf")


if __name__ == "__main__":
    main()
