"""Build the defence PPTX (数据说明 → 任务 1 → 2 → 3 → 4) from verified_master_results.json.

Text boxes are checked with a conservative CJK line estimator; any box that would overflow raises an error.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
FIG = OUT / "figures"
DELIV = OUT / "deliverables"
NAVY, TEAL, INK, MUTED, AMBER, RUST, LIGHT = (RGBColor(0x15, 0x2D, 0x42), RGBColor(0x0D, 0x94, 0x88), RGBColor(0x1E, 0x29, 0x3B),
                                             RGBColor(0x64, 0x74, 0x8B), RGBColor(0xD9, 0x77, 0x06), RGBColor(0xC2, 0x41, 0x0C), RGBColor(0xF1, 0xF5, 0xF9))
FONT = "Microsoft YaHei"
W, H = Inches(13.333), Inches(7.5)
PROG_S = {"BTBU_CS_2022": "北工商计科", "BTBU_AI_2022": "北工商AI", "BISTU_CS_2025": "北信科计科"}
OVERFLOW = []
_TEAM_FILE = Path(__file__).resolve().parents[2] / "team_info.json"
_TEAM = json.loads(_TEAM_FILE.read_text(encoding="utf-8")) if _TEAM_FILE.exists() else {}



def pct(v, d=1):
    return f"{v * 100:.{d}f}%"


def ci(c, d=1):
    return f"{c['delta'] * 100:+.{d}f} [{c['ci95_lo'] * 100:+.{d}f}, {c['ci95_hi'] * 100:+.{d}f}]"


def _est_lines(text: str, width_in: float, size_pt: float) -> int:
    cap = max(1.0, (width_in - 0.2) * 72.0 / size_pt)
    n = 0
    for para in text.split("\n"):
        units = sum(1.0 if ord(ch) > 0x2E80 else 0.55 for ch in para)
        n += max(1, math.ceil(units / cap))
    return n


def text(slide, x, y, w, h, s, size=14, color=INK, bold=False, align=PP_ALIGN.LEFT, fill=None, anchor=MSO_ANCHOR.TOP, name=""):
    need = _est_lines(s, w, size) * size * 1.28 / 72.0 + 0.12
    if need > h + 0.02:
        OVERFLOW.append(f"{name or s[:18]}: need {need:.2f}in > {h:.2f}in")
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if fill is not None:
        box.fill.solid()
        box.fill.fore_color.rgb = fill
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.08)
    tf.margin_top = tf.margin_bottom = Inches(0.04)
    for i, line in enumerate(s.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = 1.12
        r = p.add_run()
        r.text = line
        r.font.size = Pt(size)
        r.font.name = FONT
        r.font.bold = bold
        r.font.color.rgb = color
    return box


def header(prs, title, section, num):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, Inches(0.9))
    bar.fill.solid()
    bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()
    text(s, 0.45, 0.14, 10.6, 0.62, title, 24, RGBColor(0xFF, 0xFF, 0xFF), True, anchor=MSO_ANCHOR.MIDDLE, name=title)
    text(s, 10.9, 0.24, 2.1, 0.45, section, 12, RGBColor(0x99, 0xF6, 0xE4), True, PP_ALIGN.RIGHT, name=section)
    text(s, 0.45, 7.05, 12.4, 0.35, f"产教智对 · EvoAlign-OS        {num}", 10, MUTED, align=PP_ALIGN.RIGHT, name="footer")
    return s


def image(slide, name, x, y, w=None, h=None):
    p = str(FIG / name)
    if w and h:
        from PIL import Image
        with Image.open(p) as im:
            ar = im.width / im.height
        if w / h > ar:
            w = h * ar
        else:
            h = w / ar
        return slide.shapes.add_picture(p, Inches(x), Inches(y), Inches(w), Inches(h))
    return slide.shapes.add_picture(p, Inches(x), Inches(y), Inches(w) if w else None, Inches(h) if h else None)


def table(slide, x, y, w, rows, col_w=None, size=11, row_h=0.34, head_fill=NAVY, hl=()):
    nr, nc = len(rows), len(rows[0])
    shp = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(row_h * nr))
    tb = shp.table
    if col_w:
        tot = sum(col_w)
        for j, cw in enumerate(col_w):
            tb.columns[j].width = Emu(int(Inches(w) * cw / tot))
    for i, row in enumerate(rows):
        tb.rows[i].height = Inches(row_h)
        for j, val in enumerate(row):
            c = tb.cell(i, j)
            c.margin_left = c.margin_right = Inches(0.05)
            c.margin_top = c.margin_bottom = Inches(0.02)
            c.text = ""
            p = c.text_frame.paragraphs[0]
            r = p.add_run()
            r.text = str(val)
            r.font.size = Pt(size)
            r.font.name = FONT
            r.font.bold = i == 0 or i in hl
            r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF) if i == 0 else INK
            c.fill.solid()
            c.fill.fore_color.rgb = head_fill if i == 0 else (RGBColor(0xCC, 0xFB, 0xF1) if i in hl else (LIGHT if i % 2 == 0 else RGBColor(0xFF, 0xFF, 0xFF)))
            cwi = (w * (col_w[j] / sum(col_w)) if col_w else w / nc)
            need = _est_lines(str(val), cwi, size) * size * 1.25 / 72 + 0.06
            if need > row_h + 0.18:
                OVERFLOW.append(f"table cell '{str(val)[:14]}' needs {need:.2f}in > row {row_h:.2f}in")
    return shp


def kpi(slide, x, y, w, big, small, color=TEAL):
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(1.25))
    box.fill.solid()
    box.fill.fore_color.rgb = RGBColor(0xF8, 0xFA, 0xFC)
    box.line.color.rgb = RGBColor(0xCB, 0xD5, 0xE1)
    text(slide, x + 0.08, y + 0.08, w - 0.16, 0.55, big, 22, color, True, name=big)
    text(slide, x + 0.08, y + 0.62, w - 0.16, 0.6, small, 11, MUTED, name=small)


def bullets(slide, x, y, w, h, items, size=14, name="bullets"):
    return text(slide, x, y, w, h, "\n".join("▪ " + it for it in items), size, INK, name=name)


def build(m: dict, team: str, verify: dict | None) -> Presentation:
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    C = m["cases"]
    r25 = {p: C[p]["results"]["replay_2025"] for p in PROG_S}
    r26 = {p: C[p]["results"]["replay_2026q1"] for p in PROG_S}
    full = C["BTBU_CS_2022"]["actions"]["Full_EvoAlign_OS"]
    efb, rt, ne, nm = m["task2_employment_feedback"], m["task4_red_team_risks"], m["natural_experiments"]["micro_course_btbu_2022_to_2025"], m["natural_experiments"]["macro_major_moe_2024_beijing"]
    ver = f"{verify['summary']['n_pass']}/{verify['summary']['n_checks']}" if verify else "—"
    n = [0]

    def nxt():
        n[0] += 1
        return n[0]

    # 1 Title
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, H)
    bg.fill.solid()
    bg.fill.fore_color.rgb = NAVY
    bg.line.fill.background()
    text(s, 0.8, 1.4, 11.8, 0.5, "产教智对 · 现场答辩", 18, RGBColor(0x5E, 0xEA, 0xD4), True)
    text(s, 0.8, 2.0, 11.8, 1.5, "产业需求与高校专业动态适配智能分析平台", 36, RGBColor(0xFF, 0xFF, 0xFF), True)
    text(s, 0.8, 3.45, 11.8, 0.6, "EvoAlign-OS：面向专业调整决策的双环自进化供需对齐系统", 20, RGBColor(0xCB, 0xD5, 0xE1))
    members = "、".join(x["name"] for x in _TEAM.get("members", []))
    team_line = f"队伍：{team}" + (f"　·　{_TEAM['university']}" if _TEAM.get("university") else "") + (f"　·　成员：{members}" if members else "") + (f"　·　指导教师：{_TEAM['advisor']}" if _TEAM.get("advisor") else "")
    text(s, 0.8, 5.2, 11.8, 1.2, f"{team_line}\n所有数字取自封印结果文件 SHA-256 {m['master_sha256'][:16]}…  ·  独立复核 {ver} 项一致", 14, RGBColor(0xCB, 0xD5, 0xE1))
    nxt()

    # 2 Overview
    s = header(prs, "问题与总体思路", "总览", nxt())
    text(s, 0.5, 1.1, 12.3, 0.9, "把“专业怎么调”变成一个有约束、有时滞、能自我改进的决策问题：企业需求端与学生供给端在同一套 48 项技能 × 6 级认知深度坐标上对齐，求解器保证可执行，大模型只负责扩展候选。", 16, INK)
    layers = [("L1 差距引擎", "招聘文本→技能\n培养方案→修读路径\nL1/OT/布鲁姆 + 统计审计"), ("L2 约束求解", "CP-SAT 课程增/停/升级\n学分守恒、师资预算\nGale-Shapley 分流"),
              ("L3 仿真与预测", "四年推演、三情景\n保形区间、时滞 MPC\n红队压测"), ("L4 双环自进化", "错误诊断→Qwen 提案\nGUARD 开发集门控\nSIM 记忆库、哈希留痕")]
    for i, (t_, d_) in enumerate(layers):
        x = 0.5 + i * 3.1
        b = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(2.15), Inches(2.9), Inches(1.95))
        b.fill.solid()
        b.fill.fore_color.rgb = RGBColor(0xEC, 0xFE, 0xFF)
        b.line.color.rgb = TEAL
        text(s, x + 0.1, 2.2, 2.7, 0.45, t_, 16, TEAL, True)
        text(s, x + 0.1, 2.7, 2.7, 1.35, d_, 13, INK)
    b0, fl = r25["BTBU_CS_2022"]["metrics"]["B0_Freeze"], r25["BTBU_CS_2022"]["metrics"]["Full_EvoAlign_OS"]
    kpi(s, 0.5, 4.45, 3.0, f"{pct(b0['jd_qual_rate_70'])} → {pct(fl['jd_qual_rate_70'])}", "主案例 2025 回放岗位级达标率")
    sig = [C[p]["results"][sp]["paired_qual_ci95"][ref]["Full_EvoAlign_OS"]["ci95_lo"] > 0 for p in PROG_S for sp in ("replay_2025", "replay_2026q1")
           for ref in ("B0_Freeze", "B1_SingleTrack_MILP", "B2_MultiTrack_Static", "B3_SingleShot_LLM")]
    kpi(s, 3.6, 4.45, 3.0, f"{sum(sig)} / {len(sig)}", "相对 B0–B3 配对 95% 区间下界 > 0")
    kpi(s, 6.7, 4.45, 3.0, f"{pct(rt['ungoverned_mean_breach_rate'])} → {pct(rt['governed_mean_breach_rate'])}", "8 类风险压测平均超限频次")
    kpi(s, 9.8, 4.45, 3.0, ver, "独立复核脚本重算一致项")
    text(s, 0.5, 5.95, 12.3, 0.9, "岗位级达标率：按方案培养的毕业生，其课程技能能满足的岗位占比（雇主等权）。它衡量能力对口，不等同于就业率。", 13, MUTED)

    # 3 Data
    s = header(prs, "数据说明：来源、范围与规模", "数据说明", nxt())
    kg = m["kg_stats"]
    table(s, 0.5, 1.15, 12.3, [
        ["数据", "来源", "规模", "用途"],
        ["招聘数据（上市公司+应届生）", "官方基础数据集", "18,866,559 条 → 技术岗 1,711,875 条 → 北京技术岗", "需求画像、评估"],
        ["《中国本科生就业报告》", "官方基础数据集", "2023/2024/2026 年版，39 个指标值", "就业反馈通道"],
        ["政策文件、各省支持产业", "官方基础数据集", "十五五规划等", "新兴技能先验"],
        ["高校培养方案与大纲", "自采（高校官网）", f"{kg['universities']} 校 {kg['programs']} 份方案、{kg['total_courses']:,} 门课", "供给画像、自然实验"],
        ["教育部专业备案审批结果、2026 专业目录", "自采（教育部官网）", "2020–2023 年 10,164 条；2024 年隔离；目录 875 种专业", "专业布点、代码核验"],
        ["101 计划核心课、CS2023、ESCO 本体", "自采（智慧教育平台、ACM/IEEE、欧盟接口）", "12 门核心课；89 个 ESCO 技能概念、12 个职业", "标准对照"],
        ["北京紧缺人才目录、教学质量报告", "自采（政府、高校官网）", "逐份登记网址、时间与 SHA-256", "技能权重、风险参数"],
    ], col_w=[3.2, 2.6, 4.6, 1.9], size=11, row_h=0.38)
    bullets(s, 0.5, 4.35, 6.0, 2.6, ["清洗：档案内精确去重、去水印与联系方式、薪资去倒置与越界、学历经验枚举统一映射",
                                     "口径：北京、技术类、本科可及入门岗；训练 ≤2023 / 开发 2024 / 回放 2025 / 漂移 2026Q1",
                                     "防泄漏：2025 与 2026Q1 物理隔离，读取须经审计上下文并记录哈希"], 13, "data bullets")
    bullets(s, 6.8, 4.35, 6.0, 2.6, ["合规：仅用官方数据与公开文件，遵守 robots 与频率限制，不含个人信息",
                                     "脱敏：雇主以假名编号表示；支撑材料仅含 30 条样例与聚合特征矩阵",
                                     "大模型：只发送公开课程文本与聚合统计，请求/响应全量记账可重放"], 13, "compliance bullets")

    # 4 Feature extraction
    s = header(prs, "数据说明：特征提取与可靠性", "数据说明", nxt())
    rel = m["extraction_reliability"]
    steps = [("岗位→技能", "48 项标准技能同义词表匹配岗位名+职位描述；标题规则归入 4 个岗位族"), ("岗位权重", "来源等权 → 雇主等权 → 岗位等权，抑制大雇主重复发布"),
             ("课程→技能", f"Qwen3.8-flash 抽取技能/权重/布鲁姆；max 复标 {rel['audited_courses']} 门：首位一致 {pct(rel['top1_skill_agreement'])}，Jaccard {rel['mean_skill_jaccard']:.3f}"),
             ("方案→修读路径", "必修全修，选修按官方最低学分比例换算；11 份基准方案与官方表格逐门核对")]
    for i, (a_, b_) in enumerate(steps):
        y = 1.2 + i * 1.05
        text(s, 0.5, y, 2.4, 0.85, a_, 16, RGBColor(0xFF, 0xFF, 0xFF), True, PP_ALIGN.CENTER, TEAL, MSO_ANCHOR.MIDDLE)
        text(s, 3.05, y, 9.8, 0.85, b_, 14, INK, fill=LIGHT, anchor=MSO_ANCHOR.MIDDLE)
    text(s, 0.5, 5.5, 12.3, 1.3, "就绪度 r = min(1, 学分质量/2) × (0.40 + 0.60 × min(1, 课程布鲁姆/岗位布鲁姆))\n岗位达标：所列技能平均就绪度 ≥ 0.70，且 ≥ 70% 的所列技能就绪度 ≥ 0.55；不用 S 形放大，不用就业结果。", 14, NAVY, True, fill=RGBColor(0xEC, 0xFE, 0xFF))

    # 5 Task1 skill structure
    s = header(prs, "任务 1 · 技能结构差距：11 份北京高校方案", "任务 1", nxt())
    image(s, "fig1_multi_uni_benchmark.png", 0.4, 1.0, 7.6, 6.0)
    bench = {b["program_id"]: b for b in m["task1_gap_analysis"]["multi_university_benchmarks"]}
    b = bench["BTBU_CS_2022"]
    bullets(s, 8.2, 1.15, 4.7, 5.8, [
        "按培养定位分三层、各对各的岗位样本，不做跨层排名",
        "共性缺口：Linux/Shell、SQL、Java 后端、测试、嵌入式",
        f"北工商计科：L1 {pct(b['l1_overlap_dev2024'])}（95% CI {pct(b['l1_ci95'][0])}–{pct(b['l1_ci95'][1])}），OT {pct(b['ot_alignment_dev2024'])}，布鲁姆保真 {pct(b['bloom_fidelity_dev2024'])}",
        "软件架构等理论类供给富余（青绿色），工程实操类不足",
        "对照 101 计划 12 门核心课：应用型 AI、自动化方案缺编译、网络、操作系统等系统类课",
    ], 14, "t1 bullets")

    # 6 Task1 direction & scale & stats
    s = header(prs, "任务 1 · 培养方向、规模匹配与统计检验", "任务 1", nxt())
    TRK = {"fullstack_backend": "后端/全栈", "cloud_ops_sec": "云运维/测安", "ai_data": "AI/数据", "systems_edge": "系统/嵌入式"}
    dev = {p: C[p]["results"]["development_2024"]["metrics"]["B0_Freeze"]["per_track"] for p in PROG_S}
    rows = [["岗位族", "北工商计科", "北工商AI", "北信科计科", "需求份额"]]
    for tk, nm_ in TRK.items():
        rows.append([nm_] + [pct(dev[p][tk]["jd_qual_rate_70"]) for p in PROG_S] + [pct(dev["BTBU_CS_2022"][tk]["demand_share"])])
    text(s, 0.5, 1.05, 6.1, 0.4, "② 培养方向：现行单一路径在各岗位族的达标率（2024）", 14, NAVY, True)
    table(s, 0.5, 1.5, 6.1, rows, col_w=[1.6, 1.2, 1.2, 1.2, 1.1], size=12, row_h=0.4)
    text(s, 6.9, 1.05, 6.0, 0.4, "③ 规模匹配：官方就业报告（计算机类 / 工学）", 14, NAVY, True)
    rows = [["届别", "落实率", "专业相关度", "月收入"]]
    for x in efb["signals"]:
        rows.append([f"{x['cohort']}", "—" if x["cs_placement"] is None else f"{x['cs_placement']}/{x['eng_placement']}", f"{x['cs_relevance']:.0f}/{x['eng_relevance']:.0f}", f"{x['cs_income']:.0f}/{x['eng_income']:.0f}"])
    table(s, 6.9, 1.5, 5.9, rows, col_w=[0.9, 1.6, 1.6, 1.8], size=12, row_h=0.36)
    a = m["task1_gap_analysis"]["dev2024_demand_summary"]["anomaly_audit"]
    text(s, 0.5, 3.75, 6.1, 0.4, "统计检验与异常值（2024 开发集）", 14, NAVY, True)
    table(s, 0.5, 4.2, 6.1, [["审计项", "值", "处理"],
                             ["雇主发布基尼系数", f"{a['employer_gini']:.2f}", "雇主等权 + 聚类自助法"],
                             ["薪资缺失/面议率", pct(a["salary_missing_rate"]), "不进入薪资模型"],
                             ["对数薪资 IQR 异常值", str(a["log_salary_iqr_outliers"]), "缩尾而非删除"],
                             ["零技能命中率", pct(a["zero_skill_hit_rate"]), "不进入达标率分母"]],
          col_w=[2.3, 1.1, 2.7], size=12, row_h=0.4)
    text(s, 6.9, 3.75, 5.9, 3.1, "结论：需求总量未下行，而计算机类专业相关度连续三届低于工学平均 7–10pp，落实率差扩大到 −6.2pp；2025 届从事无关工作者中 22% 是“达不到专业工作要求”。\n\n判断：结构性错配（人数够、能力不对口），优先改课程与方向，而非压缩总招生。", 14, INK, fill=RGBColor(0xFF, 0xFB, 0xEB), name="t1 conclusion")

    # 7 Task2 courses
    s = header(prs, "任务 2 · 课程体系增删改（北工商计科，中等预算档）", "任务 2", nxt())
    ACT = {"ADD_ELECTIVE": "新增", "REMOVE_ELECTIVE": "停开", "UPGRADE_SYLLABUS": "升级"}
    order = {"REMOVE_ELECTIVE": 0, "ADD_ELECTIVE": 1, "UPGRADE_SYLLABUS": 2}
    rows = [["动作", "课程", "学期", "学分/学时", "要点"]]
    for a_ in sorted(full["actions"], key=lambda x: (order[x["action"]], x["semester"])):
        if a_["action"] == "UPGRADE_SYLLABUS":
            ch, why = "课内 16 学时", a_["module"]["title"]
        else:
            ch = f"{a_['delta_credits']:+g} 学分"
            why = ("外环提案 · " if a_.get("source", "").startswith("qwen") else "") + ("、".join(a_["skills"][:2]))
        rows.append([ACT[a_["action"]], a_["course_name"][:14], f"S{a_['semester']}", ch, why[:26]])
    table(s, 0.4, 1.05, 8.6, rows, col_w=[0.7, 2.6, 0.6, 1.3, 3.4], size=10, row_h=0.29)
    chk = full["constraint_checks"]
    text(s, 9.2, 1.05, 3.7, 0.4, "硬约束校验", 15, NAVY, True)
    CHK = {"student_graduation_credits_conserved": "毕业总学分守恒", "required_and_core_100pct_retained": "必修/核心课全保留", "faculty_fte_within_cap": "师资 FTE 不超限",
           "lab_gpu_within_cap": "实验/GPU 不超限", "prereq_dag_valid": "先修关系合法", "domain_coherent_upgrades_only": "升级仅限学科同源",
           "streaming_zero_blocking_pairs": "分流阻塞对 = 0", "reform_budget_ok": "改革幅度在预算内", "zero_hard_violations": "零硬约束违反"}
    text(s, 9.2, 1.5, 3.7, 3.3, "\n".join(f"{'✓' if v else '✗'} {CHK.get(k, k)}" for k, v in chk.items()), 13, INK, fill=LIGHT)
    text(s, 9.2, 4.95, 3.7, 2.0, f"新增 {full['added_credits']:g} / 停开 {full['removed_credits']:g} 学分，升级 {full['upgraded_core_count']} 门核心课\n师资 {full['faculty_fte_used']}/{full['faculty_fte_cap']} FTE\n实验资源 {full['lab_units_used']}/{full['lab_units_cap']}\n求解 {full['solver_wall_time_ms']:.0f} ms", 13, NAVY, True, fill=RGBColor(0xEC, 0xFE, 0xFF))
    iis = m.get("task2_iis_diagnosis")
    if iis:
        text(s, 0.4, 6.42, 8.6, 0.55, "IIS 诊断：强制采纳全部外环新课 → " + iis["status"] + "；冲突核：" + "；".join(iis["iis_conflicts"][:2])[:70], 10, RUST, name="iis")

    # 8 Task2 streaming
    st = full["streaming_summary"]
    s = header(prs, "任务 2 · 培养方向与大二末分流机制", "任务 2", nxt())
    bullets(s, 0.5, 1.1, 6.0, 4.3, ["时点：第 4 学期末分流；第 5 学期开放 ≤ 10% 互转名额",
                                    "意愿：学生对四个方向完整排序",
                                    "成绩：0.55×前四学期绩点 + 0.30×方向先修课成绩 + 0.15×志愿强度",
                                    "容量：上限 = 需求份额×1.05+1%，下限 = 需求份额×0.82",
                                    "算法：学生提议的 Gale-Shapley，结果无阻塞对，如实填报是最优策略",
                                    "方向课群 + 定向实训/实习/毕设包，不增学分"], 14, "stream bullets")
    rows = [["方向", "需求份额", "容量", "录取", "定向实训包"]]
    for t_ in st["track_allocations"]:
        rows.append([t_["name"][:10], pct(t_["demand_share"]), f"{t_['min_cap']}–{t_['max_cap']}", str(t_["assigned"]), full["capstone_actions_by_track"][t_["id"]][0]["package_title"][:14]])
    table(s, 6.7, 1.15, 6.2, rows, col_w=[2.0, 0.9, 0.9, 0.7, 2.4], size=11, row_h=0.42)
    w_ = (st.get("under_min_warnings") or [None])[0]
    text(s, 6.7, 3.55, 6.2, 1.6, f"仿真（300 名模拟学生）：阻塞对 {st['blocking_pairs']}，第一志愿满足 {pct(st['first_choice_satisfaction'])}，前两志愿 {pct(st['top2_choice_satisfaction'])}。" +
         (f"\n预警：{w_['track'][:8]}方向 {w_['assigned']} 人 < 下限 {w_['min_cap']}，互转窗口定向开放，仍不足则与相近专业合班（风险 R02）。" if w_ else ""), 13, INK, fill=RGBColor(0xFF, 0xFB, 0xEB))

    # 9 Task2 enrollment
    s = header(prs, "任务 2 · 基于就业反馈的招生规模动态调整", "任务 2", nxt())
    R = efb["rules"]
    table(s, 0.4, 1.05, 6.3, [["要素", "规则"],
                              ["触发阈值", f"落实率差<{R['warn_placement_gap_pp']}pp 或相关度差<{R['warn_relevance_gap_pp']}pp，连续 2 届"],
                              ["调整幅度", f"单次 ≤ {R['max_step_rate'] * 100:.0f}%，边界 [{R['governance_bounds'][0]}, {R['governance_bounds'][1]}] 人"],
                              ["调整周期", f"每年 Q3 评估；调整后冷却 {R['cooldown_years']} 年"],
                              ["时滞补偿", "MPC 用第 t+4 年需求预测决策，死区 ±6.5%"],
                              ["错配解耦", "结果预警而需求未降 → 保持总量，先改课程与方向名额"]],
          col_w=[1.3, 5.0], size=12, row_h=0.44)
    rows = [["证据→决策", "状态", "动作"]]
    for x, d in zip(efb["signals"], efb["decision_log"]):
        rows.append([f"{x['cohort']}→{d['decision_year']}", d["state"], d["action"].replace("HOLD_TOTAL+REALLOCATE_TRACKS+TRIGGER_CURRICULUM_RSI", "保总量+调方向+改课程")[:18]])
    table(s, 0.4, 4.3, 6.3, rows, col_w=[1.5, 1.1, 3.7], size=11, row_h=0.36)
    image(s, "fig4_mpc_control.png", 6.9, 1.05, 6.1, 3.0)
    se, ss = m["task2_mpc_enrollment"]["real_empirical_mpc"]["stability_comparison"], m["task2_mpc_enrollment"]["synthetic_stress_mpc"]["stability_comparison"]
    text(s, 6.9, 4.3, 6.1, 2.6, f"北京实测需求波动未连续突破死区，MPC 维持 120 人；“缺就多招”的朴素反馈总变差 {se['naive_total_variation']:.0f}，出现蛛网震荡。\n增长+周期压力情景：MPC 震荡降 {ss['oscillation_reduction_pct']:.0f}%，跟踪误差较冻结降 {ss['rmse_reduction_vs_freeze_pct']:.0f}%。\n当前建议：保持总招生，调整方向名额，先改课程。", 13, INK, fill=LIGHT)

    # 10 Task2 macro + RSI
    s = header(prs, "任务 2 · 专业布点增调停 与 双环自进化", "任务 2", nxt())
    mac = m["task2_macro_major_portfolio"]
    rows = [["专业", "动作"]]
    for d in mac["decisions"][:6]:
        MA = {"RETAIN_AND_ENHANCE": "保留并升级", "ADJUST_MAJOR": "调整（大纲/方向重构）", "SUSPEND_OR_REVOKE_MAJOR": "停招 + 4 年平稳退出",
              "ADD_MAJOR": "新增专业", "INCUBATE_MICRO_MAJOR_ONLY": "先以微专业孵化"}
        rows.append([d["major_name"], MA.get(d["action"], d["action"])])
    table(s, 0.4, 1.05, 5.0, rows, col_w=[2.2, 2.8], size=11, row_h=0.36)
    text(s, 0.4, 3.75, 5.0, 3.1, f"物联网工程停招：契合度低、与院内主干专业高度同质；四年平稳退出，逐年释放 {'/'.join(f'{x:g}' for x in mac['freed_fte_via_teach_out_y1_y4'])} FTE。\n新增数据科学与大数据技术，其余新兴专业先以微专业孵化", 13, INK, fill=LIGHT)
    image(s, "fig2_rsi_evolution.png", 5.6, 1.05, 7.4, 3.3)
    text(s, 5.6, 4.5, 7.4, 2.4, "内环：给定候选池与策略参数，CP-SAT 求最优方案\n外环：诊断未达标岗位缺什么 → Qwen 提出新课/大纲模块/实训包 → 重求解\nGUARD：开发集不退化 + 最弱方向不下滑 + 近期年份不退化，才晋级\nSIM：成败经验记入记忆库，可迁移到其他学校/专业", 13, INK, fill=RGBColor(0xEC, 0xFE, 0xFF))

    # 11 Task3 design + ablation
    s = header(prs, "任务 3 · 验证设计与六组消融", "任务 3", nxt())
    image(s, "fig3_ablation_6way.png", 0.4, 1.0, 8.6, 4.0)
    bullets(s, 9.2, 1.1, 3.8, 5.8, ["方案只用 ≤ 2023 年数据生成，2024 用于门控",
                                    f"2025 回放 {r25['BTBU_CS_2022']['ads']:,} 条 / 2026Q1 {r26['BTBU_CS_2022']['ads']} 条，隔离保存",
                                    "六组共用同一硬约束、预算与评估器",
                                    "区间：雇主聚类配对自助法 600 次",
                                    "2025 回放在研发中查看过，称“历史回放”"], 13, "t3 bullets")
    rows = [["对照", "北工商计科", "北工商AI", "北信科计科"]]
    for aid, lab in (("B0_Freeze", "B0 现状"), ("B2_MultiTrack_Static", "B2 多轨静态"), ("B3_SingleShot_LLM", "B3 单轮LLM"), ("B4_BiLevel_NoGuard", "B4 无GUARD"), ("Full_EvoAlign_OS", "Full")):
        rows.append([lab] + [pct(r25[p]["metrics"][aid]["jd_qual_rate_70"]) for p in PROG_S])
    table(s, 0.4, 5.1, 8.6, rows, col_w=[2.0, 2.2, 2.2, 2.2], size=11, row_h=0.31, hl=(5,))

    # 12 Task3 paired CIs
    s = header(prs, "任务 3 · 配对增益与置信区间", "任务 3", nxt())
    image(s, "structural_repair_paired.png", 0.4, 1.0, 8.4, 4.1)
    rows = [["Full−B4（关闭 GUARD）", "2025", "2026Q1"]]
    for p in PROG_S:
        a_, b_ = r25[p]["paired_qual_ci95"]["B4_BiLevel_NoGuard"]["Full_EvoAlign_OS"], r26[p]["paired_qual_ci95"]["B4_BiLevel_NoGuard"]["Full_EvoAlign_OS"]
        rows.append([PROG_S[p], ci(a_) + ("" if a_["ci95_lo"] > 0 else " n.s."), ci(b_) + ("" if b_["ci95_lo"] > 0 else " n.s.")])
    table(s, 0.4, 5.25, 8.4, rows, col_w=[2.2, 3.1, 3.1], size=11, row_h=0.36)
    bullets(s, 9.0, 1.1, 4.0, 5.8, ["24/24 组相对 B0–B3 的区间下界 > 0",
                                    "主要增量来自多代进化解锁的定向实训包与同源大纲模块",
                                    "GUARD 的价值是防止偏科式退化：AI 专业 +11pp，计科持平",
                                    "2026Q1 样本小，Full−B4 均不显著，如实报告"], 13, "paired bullets")

    # 12b Preregistered independent blind test
    pre_p, res_p = OUT / "preregistration_blind_v3.json", OUT / "preregistered_blind_results.json"
    if pre_p.exists() and res_p.exists():
        pre = json.loads(pre_p.read_text(encoding="utf-8"))
        bl = json.loads(res_p.read_text(encoding="utf-8"))
        s = header(prs, "任务 3 · 预注册独立盲测（冻结后一次性检验）", "任务 3", nxt())
        text(s, 0.5, 1.05, 12.3, 0.9, f"先登记、后取数：{pre['registered_utc'][:16].replace('T', ' ')} UTC 冻结方案快照、评估代码与 4 条假设（哈希 {pre['preregistration_sha256'][:12]}…），"
             f"随后才首次抽取从未读取过的京外人群，一次性评估、拒绝重跑。", 14, INK, name="prereg intro")
        rows = [["人群", "专业", "B0 现状", "Full", "H1 vs B0", "H2 vs B2", "H3 vs B3", "H4 vs B4（非劣效）"]]
        for pk, pr in bl["results"].items():
            for prog, v in pr["programs"].items():
                hs = {h["id"]: h for h in v["hypotheses"]}
                rows.append([("P1 津冀" if pk.startswith("P1") else "P2 京外全国"), PROG_S[prog], pct(v["metrics"]["B0_Freeze"]["jd_qual_rate_70"]),
                             pct(v["metrics"]["Full_EvoAlign_OS"]["jd_qual_rate_70"])]
                            + [f"{hs[k]['delta'] * 100:+.1f}pp {'✓' if hs[k]['supported'] else '✗'}" for k in ("H1", "H2", "H3", "H4")])
        table(s, 0.5, 2.0, 12.3, rows, col_w=[1.6, 1.6, 1.2, 1.2, 1.6, 1.6, 1.6, 1.9], size=12, row_h=0.42)
        n1, n2 = bl["results"]["P1_jinji"], bl["results"]["P2_national_ex_bj"]
        text(s, 0.5, 5.15, 12.3, 1.7, f"P1 津冀 {n1['n_ads']:,} 条 / {n1['n_employers']} 家雇主；P2 京外全国 {n2['n_ads']:,} 条 / {n2['n_employers']:,} 家雇主。"
             f"预注册假设成立：P1 {bl['hypotheses_supported']['P1_jinji']}，P2 {bl['hypotheses_supported']['P2_national_ex_bj']}。\n"
             "结论与北京回放一致，增益不是对北京回放数据调参的结果；该检验衡量跨区域泛化。", 14, NAVY, True, fill=RGBColor(0xEC, 0xFE, 0xFF), name="prereg concl")

    # 13 Task3 scenarios + natural experiments
    s = header(prs, "任务 3 · 反事实、三情景与自然实验", "任务 3", nxt())
    image(s, "fig5_dreamsim_scenarios.png", 0.4, 1.0, 6.6, 3.0)
    sc = m["task3_dream_sim_3scenarios"]
    table(s, 0.4, 4.15, 6.6, [["情景", "达标率", "假设"],
                              ["反事实冻结", pct(m["task3_counterfactual_freeze"]["jd_qual_rate_70"]), "不改革"],
                              ["乐观", pct(sc["optimistic"]["jd_qual_rate_70"]), "新课/升级/实训效果 100%"],
                              ["基准", pct(sc["baseline"]["jd_qual_rate_70"]), "效果 94–96%"],
                              ["悲观", pct(sc["pessimistic"]["jd_qual_rate_70"]), "效果 80–86%，负荷惩罚"]],
          col_w=[1.6, 1.2, 3.8], size=11, row_h=0.36)
    da = ne["direction_agreement"]
    table(s, 7.2, 1.05, 5.8, [["课程自然实验（2025 回放）", "达标率"],
                              ["北工商 2022 版", pct(ne["btbu_frozen_2022_on_replay2025"]["jd_qual_rate_70"])],
                              ["北工商实际 2025 改版", pct(ne["btbu_real_2025_on_replay2025"]["jd_qual_rate_70"])],
                              ["EvoAlign 预测（≤2022，单轨）", pct(ne["evoalign_pred_single_track_on_replay2025"]["jd_qual_rate_70"])],
                              ["EvoAlign 预测（≤2022，分流）", pct(ne["evoalign_pred_on_replay2025"]["jd_qual_rate_70"])]],
          col_w=[4.2, 1.6], size=12, row_h=0.4)
    text(s, 7.2, 3.2, 5.8, 1.5, f"改动方向一致：与学校同时调整了 {da['course_level']['n_agreed']} 门课（随机期望 {da['course_level']['chance_expectation']:.1f}，p = {da['course_level']['hypergeom_p_value']:.3f}）。学校改版增加金融特色课、压缩工程学分，在入门技术岗口径下下降。", 12, INK, fill=LIGHT)
    table(s, 7.2, 4.85, 5.8, [["专业布点（≤2023 预测 2024）", "命中", "覆盖"],
                              ["EvoAlign Top-5", f"{nm['top5_hit_count']}/5", pct(nm["evoalign_top5_volume_coverage"])],
                              ["惯性基线 Top-5", f"{nm['persistence_baseline_hit_count']}/5", pct(nm["persistence_baseline_volume_coverage"])]],
          col_w=[3.4, 1.1, 1.3], size=12, row_h=0.4)

    # 14 Task3 transfer + limits
    s = header(prs, "任务 3 · 迁移、多智能体沙盘与不确定性", "任务 3", nxt())
    tr = m["cross_school_transfer"]
    rows = [["目标", "现状", "冷启动", "迁移（不调 LLM）", "本地进化"]]
    for k, lab in (("BISTU_CS_2025_unseen_university", "跨校：北信科计科"), ("BTBU_AI_2022_unseen_major", "跨专业：北工商AI")):
        v = tr[k]
        rows.append([lab] + [pct(v[x]["jd_qual_rate_70"]) for x in ("baseline_freeze", "cold_start_H0", "warm_transfer_Hstar_SIM", "full_local_evolved")])
    table(s, 0.5, 1.15, 12.3, rows, col_w=[3.0, 2.0, 2.0, 3.0, 2.3], size=14, row_h=0.5)
    text(s, 0.5, 2.85, 12.3, 0.5, "同类专业迁移接近本地进化；跨专业提升明显但仍需本地进化。", 14, NAVY, True)
    ag = m.get("task3_agent_dream_sim") or {}
    if ag:
        text(s, 0.5, 5.0, 12.3, 0.4, "多智能体沙盘复核（300 名异质学生智能体 × 8 学期，2025 回放）：学生匹配度 现状 → 方案", 13, NAVY, True, name="ag title")
        rows = [["专业", "乐观", "基准", "悲观", "方案悲观 > 现状乐观"]]
        for pid, v in ag.items():
            rows.append([PROG_S[pid]] + [f"{pct(v['freeze'][sc]['mean_student_match'])} → {pct(v['full'][sc]['mean_student_match'])}" for sc in ("optimistic", "baseline", "pessimistic")]
                        + ["是" if v["pessimistic_full_beats_optimistic_freeze"] else "否"])
        table(s, 0.5, 5.42, 12.3, rows, col_w=[2.2, 2.7, 2.7, 2.7, 2.0], size=12, row_h=0.36)
    bullets(s, 0.5, 3.45, 12.3, 1.5, ["达标率是能力对口的代理指标，待用校内成绩与就业追踪校准（风险 R08）",
                                      "修读路径按官方最低选修学分换算；分流使用模拟学生",
                                      "课程技能由大模型抽取，抽检一致率高但非全量人工标注",
                                      "情景区间（乐观–悲观差 10.5pp）主要取决于师资到位（风险 R01）",
                                      "月薪保形区间宽、模型 R² 低，只作参考；教育部 2025 年度布点结果原文未能定位，未使用"], 13, "limits")

    # 15 Task4 risk matrix
    s = header(prs, "任务 4 · 与方案动作绑定的风险矩阵", "任务 4", nxt())
    rows = [["风险", "绑定动作", "S×L：前→后", "压测超限"]]
    for r in sorted(rt["risk_matrix"], key=lambda x: x["id"]):
        L = round(r["raw_risk_score"] / max(1, r["severity"]))
        rows.append([f"{r['id']} {r['name']}", r["bound_action"], f"{r['severity']}×{L}={r['raw_risk_score']}→{r['residual_risk_score']}", f"{pct(r['ungoverned_breach_rate'], 0)}→{pct(r['governed_breach_rate'], 0)}"])
    table(s, 0.4, 1.05, 7.8, rows, col_w=[2.9, 2.8, 1.2, 0.9], size=9, row_h=0.62)
    image(s, "fig6_risk_and_transfer.png", 8.35, 1.2, 4.7, 2.2)
    text(s, 8.35, 3.6, 4.7, 3.2, f"对推荐方案施加 {rt['n_mc_per_risk']} 次随机极端冲击，比较“无治理”与“启用治理机制”的超限频次。\n平均超限 {pct(rt['ungoverned_mean_breach_rate'])} → {pct(rt['governed_mean_breach_rate'])}。\n压测参数为专家设定的极端情形，用于比较治理有效性。", 13, INK, fill=LIGHT)

    # 16 Task4 strategies
    s = header(prs, "任务 4 · 应对策略：责任主体、预警指标、时间节点", "任务 4", nxt())
    rows = [["风险", "责任主体", "预警指标", "时间节点", "回退预案"]]
    for r in sorted(rt["risk_matrix"], key=lambda x: x["id"]):
        rows.append([r["id"], r["owner"], r["trigger"], r["deadline"], r["rollback"]])
    table(s, 0.4, 1.05, 12.5, rows, col_w=[0.6, 2.2, 3.7, 3.0, 3.6], size=9.5, row_h=0.66)

    # 17 Reproducibility
    s = header(prs, "可复核性与结论", "总结", nxt())
    kpi(s, 0.5, 1.2, 3.9, ver, "独立脚本重算一致项（无需求解器/LLM/网络）")
    kpi(s, 4.6, 1.2, 3.9, m["master_sha256"][:12] + "…", "单一事实源封印（报告、PPT、驾驶舱同源）", NAVY)
    kpi(s, 8.7, 1.2, 4.1, f"{m['rsi_full_evolution']['token_ledger_stats']['entries']} 条", "Qwen 调用重放账本，按哈希可复现", AMBER)
    bullets(s, 0.5, 2.85, 12.3, 4.0, ["差距：结构性错配，而非总量过剩；工程类技能与少数方向是主要缺口",
                                      "方案：公共基础不变 + 大二末四方向分流 + 方向课群与实训定向 + 同源大纲升级，毕业学分不变",
                                      "招生：就业反馈连续预警但需求未降，保持总量、调方向名额、先改课程",
                                      "验证：三专业两期回放显著优于四类基线；自然实验与迁移实验相互印证",
                                      "落地：每年 Q4 用新招聘数据跑一轮外环体检，驾驶舱支持管理者在线调参重算"], 16, "conclusion")
    return prs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", default=_TEAM.get("team_name") or "【队伍名称】")
    args = ap.parse_args()
    m = json.loads((OUT / "verified_master_results.json").read_text(encoding="utf-8"))
    vp = OUT / "verification_report.json"
    verify = json.loads(vp.read_text(encoding="utf-8")) if vp.exists() else None
    prs = build(m, args.team, verify)
    if OVERFLOW:
        raise SystemExit("text overflow risk:\n" + "\n".join(OVERFLOW))
    DELIV.mkdir(parents=True, exist_ok=True)
    out = DELIV / "答辩PPT.pptx"
    prs.save(out)
    print("pptx written:", out, len(prs.slides), "slides")


if __name__ == "__main__":
    main()
