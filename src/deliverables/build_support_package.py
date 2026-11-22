"""Assemble the ≤ 50 MB data-support package and the final submission ZIP.

Layout (per 《作品提交模板包》第六部分):
  {team}-方案报告/{team}_方案报告.pdf
  {team}-数据支撑材料/...                (README, 登记表, 核对表, 输入/输出样例, 脚本, 复核数据)
  {team}-演示视频/{team}_演示视频.mp4
  {team}-答辩PPT/{team}_答辩PPT.pptx
Run from the competition root:  python evoalign_os/src/deliverables/build_support_package.py --team 队伍名称
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
DELIV = OUT / "deliverables"
PROC = ROOT / "data" / "processed"
VER = ROOT / "data" / "verification"
_TEAM_FILE = Path(__file__).resolve().parents[2] / "team_info.json"
_TEAM = json.loads(_TEAM_FILE.read_text(encoding="utf-8")) if _TEAM_FILE.exists() else {}
SECRET = re.compile(rb"sk-ws-[A-Za-z0-9._-]{8,}|sk-[A-Za-z0-9]{24,}|DASHSCOPE_API_KEY\s*=\s*\S+|QWEN_API_KEY\s*=\s*\S+")
PROC_KEEP = ("dynamic_kg.json", "university_curricula_enriched.json", "university_programs_summary.json", "employment_feedback_series.json",
             "moe_major_lifecycle_2021_2023.json", "moe_major_lifecycle_2020_2023.json", "moe_undergrad_catalog_2026.json",
             "cs101_core_courses.json", "esco_alignment.json", "beijing_talent_shortage_catalog.json", "harvest_manifest.json",
             "demand_slices_manifest.json", "beijing_university_quality_governance.json")
OUT_KEEP = ("verified_master_results.json", "plan_snapshots.json", "verification_report.json", "replay_ledger.jsonl",
            "blind_access_audit.jsonl", "task1_gap_analysis_dev.json",
            "preregistration_blind_v3.json", "preregistration_blind_v3.withdrawn_1.json", "preregistered_blind_results.json",
            "macro_rolling_2025_partial.json")
REQS = """numpy==2.4.4
scipy==1.17.1
scikit-learn==1.8.0
pandas==3.0.2
pyarrow==26.0.0
duckdb==1.5.0
ortools==9.15.6755
fastapi==0.165.0
uvicorn==0.52.0
pydantic==2.13.4
matplotlib==3.10.8
pillow==13.0.0
beautifulsoup4==4.15.0
openpyxl==3.1.5
# 仅重新生成交付物/重新调用大模型时需要：
openai==3.57.0
python-pptx==1.0.2
playwright==1.62.0
edge-tts==7.3.1
"""


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def style(ws, widths):
    hdr = PatternFill("solid", fgColor="152D42")
    for c in ws[1]:
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = hdr
        c.alignment = Alignment(vertical="center", wrap_text=True)
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(vertical="top", wrap_text=True)
    ws.freeze_panes = "A2"


def register_xlsx(path: Path):
    wb = Workbook()
    ws = wb.active
    ws.title = "自采数据登记表"
    ws.append(["序号", "数据名称", "来源渠道（含网址）", "采集时间", "采集方式", "条数/规模", "合规性自查结论", "类别", "文件 SHA-256", "状态"])
    with (ROOT / "data" / "self_collected_register.csv").open(encoding="utf-8") as f:
        for i, r in enumerate(csv.DictReader(f), 1):
            scale = f"{int(r['文件字节'] or 0) / 1024:.0f} KB；{r['提取页数'] or '—'} 页；{int(r['提取字符数'] or 0):,} 字" if r["状态"] == "ok" else "未获取"
            ws.append([i, f"{r['来源机构']}《{r['数据名称']}》（{r['年份']}）", f"{r['来源机构']}官网公开文件：{r['来源URL']}",
                       r["采集时间UTC"], "公开页面/文件下载，程序解析（限速、单次获取）", scale,
                       "已核对来源为政府/高校/国际组织依法公开文档；仅提取课程/目录/统计指标，未采集个人信息",
                       r["类别"], r["SHA256"], r["状态"]])
    style(ws, [6, 46, 60, 20, 26, 26, 44, 18, 30, 8])
    ws2 = wb.create_sheet("官方基础数据使用")
    ws2.append(["数据集", "使用范围", "处理", "用途"])
    for row in (["上市公司招聘大数据 2014-2026.3 + 应届生招聘大数据", "原始 18,866,559 条；技术候选 1,711,875 条；北京技术岗 2018–2026Q1",
                 "档案内精确去重、去水印与联系方式、薪资清洗、学历经验映射、雇主假名化", "需求画像、评估、招生需求指数"],
                ["本科生就业报告 2023/2024/2026", "39 个指标值（计算机类/工学/人工智能）", "表格标题+行精确定位并断言校验", "就业反馈通道"],
                ["政策文件、各省支持产业", "十五五规划及专项规划等", "文本检索", "新兴技能与方向先验"],
                ["工商数据（抽样版）", "按数据字典建议去除个体工商户", "仅背景统计", "未进入主模型"]):
        ws2.append(row)
    style(ws2, [40, 50, 50, 30])
    wb.save(path)


def results_xlsx(path: Path, m: dict, verify: dict):
    wb = Workbook()
    ws = wb.active
    ws.title = "六组消融"
    ws.append(["专业", "回放期", "对照组", "岗位数", "岗位级达标率", "平均就绪度", "最弱方向就绪度", "方向 L1", "方向 OT"])
    for pid, c in m["cases"].items():
        for sp, res in c["results"].items():
            for arm, mm in res["metrics"].items():
                ws.append([pid, sp, arm, res["ads"], mm["jd_qual_rate_70"], mm["jd_mean_readiness"], mm["worst_track_readiness"], mm["track_l1_overlap"], mm["track_ot_alignment"]])
    style(ws, [16, 18, 24, 9, 14, 12, 16, 10, 10])
    ws = wb.create_sheet("配对区间")
    ws.append(["专业", "回放期", "对照", "Full−对照", "95%下界", "95%上界", "显著(下界>0)", "单侧p"])
    for pid, c in m["cases"].items():
        for sp, res in c["results"].items():
            for ref, d in res["paired_qual_ci95"].items():
                x = d["Full_EvoAlign_OS"]
                ws.append([pid, sp, ref, x["delta"], x["ci95_lo"], x["ci95_hi"], x["significant_positive"], x["p_boot_one_sided"]])
    style(ws, [16, 18, 24, 12, 12, 12, 14, 10])
    ws = wb.create_sheet("任务1基准")
    ws.append(["方案", "高校", "定位岗位样本", "对应岗位L1", "对应岗位OT", "入门岗L1", "L1 95%下界", "L1 95%上界", "布鲁姆保真", "学分敏感性150/160/170", "缺口Top5"])
    for b in m["task1_gap_analysis"]["multi_university_benchmarks"]:
        ws.append([b["program_id"], b["university"], b["role_target"], b["matched_l1_overlap"], b["matched_ot_alignment"], b["l1_overlap_dev2024"],
                   b["l1_ci95"][0], b["l1_ci95"][1], b["bloom_fidelity_dev2024"], ", ".join(map(str, b["matched_l1_sensitivity_150_160_170"])),
                   "；".join(f"{x[1]}({x[2]:+.3f})" for x in b["top5_deficits"])])
    style(ws, [18, 20, 12, 12, 12, 10, 10, 10, 10, 22, 70])
    ws = wb.create_sheet("课程动作_北工商计科")
    ws.append(["动作", "课程", "学期", "学分", "学分变化", "技能", "面向方向", "来源", "理由/大纲模块"])
    for a in m["cases"]["BTBU_CS_2022"]["actions"]["Full_EvoAlign_OS"]["actions"]:
        ws.append([a["action"], a["course_name"], a["semester"], a["credits"], a.get("delta_credits"), ", ".join(a["skills"]), ", ".join(a.get("target_tracks", [])),
                   a.get("source", ""), (a["module"]["title"] + "；" + a["module"]["assessment"]) if a.get("module") else a.get("reason", "")])
    style(ws, [18, 30, 6, 6, 9, 40, 30, 12, 60])
    ws = wb.create_sheet("风险矩阵")
    ws.append(["编号", "风险", "绑定动作", "影响对象", "严重度S", "治理前风险分", "治理后风险分", "无治理超限频次", "治理后超限频次", "责任主体", "预警指标", "时间节点", "应对", "回退"])
    for r in m["task4_red_team_risks"]["risk_matrix"]:
        ws.append([r["id"], r["name"], r["bound_action"], r["affected_group"], r["severity"], r["raw_risk_score"], r["residual_risk_score"], r["ungoverned_breach_rate"],
                   r["governed_breach_rate"], r["owner"], r["trigger"], r["deadline"], r["mitigation"], r["rollback"]])
    style(ws, [6, 30, 30, 26, 8, 10, 10, 10, 10, 24, 34, 30, 36, 36])
    ws = wb.create_sheet("就业反馈与招生")
    efb = m["task2_employment_feedback"]
    ws.append(["证据届别", "决策年", "落实率差(pp)", "相关度差(pp)", "收入相对差", "状态", "动作", "招生数"])
    for s, d in zip(efb["signals"], efb["decision_log"]):
        ws.append([s["cohort"], d["decision_year"], s["gap_placement_pp"], s["gap_relevance_pp"], s["gap_income_rel"], d["state"], d["action"], d["intake_after"]])
    ws.append([])
    ws.append(["年份", "需求指数", "当年需求目标", "MPC招生", "MPC毕业供给", "朴素招生", "朴素毕业供给", "MPC动作"])
    for i, h in enumerate(m["task2_mpc_enrollment"]["real_empirical_mpc"]["history"]):
        ws.append([2018 + i, h["demand_index"], h["target_capacity"], h["mpc_intake"], h["mpc_graduates"], h["naive_intake"], h["naive_graduates"], h["mpc_action"]])
    style(ws, [10, 10, 12, 12, 12, 10, 40, 10])
    ws = wb.create_sheet("独立复核结果")
    ws.append(["复核组", "项目", "重算值", "封印值", "一致"])
    for c in verify["checks"]:
        ws.append([c["group"], c["item"], str(c["recomputed"]), str(c["sealed"]), c["pass"]])
    style(ws, [12, 60, 24, 24, 8])
    wb.save(path)


def readme(m: dict, verify: dict, team: str) -> str:
    s = verify["summary"]
    return f"""# {team} · 数据支撑材料（EvoAlign-OS）

本材料用于核对方案报告中的全部关键结果。所有报告、PPT、视频与驾驶舱数字取自同一份封印文件
`outputs/verified_master_results.json`（SHA-256 `{m['master_sha256']}`）。

## 1. 一步复核（约 3 秒，无需求解器、大模型或网络）

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python src/experiments/verify_key_results.py --root .
```

脚本读取封印结果、`outputs/plan_snapshots.json`（各对照组逐方向课程清单）与 `data/verification/`（脱敏逐岗位特征矩阵），
重新计算并逐项比对，输出 `outputs/verification_report.json`。当前结果：**{s['n_pass']}/{s['n_checks']} 项一致**。

| 复核组 | 内容 |
|---|---|
| 0 封印 | 结果文件 SHA-256、课程快照与结果文件的对应 |
| 1 消融 | 3 专业 × 3 期 × 6 组的达标率、就绪度、最弱方向、L1、OT |
| 2 配对区间 | Full 相对 B0–B4 的雇主聚类配对自助法 95% 区间（600 次，种子 42） |
| 3 自然实验 / 帕累托 | 北工商 2022/2025 版与预测方案、三档预算方案的 2025 回放达标率 |
| 4 任务1 | 11 份方案的 L1 与 OT（Wasserstein-1 精确线性规划） |
| 5 招生调控 | 2018–2024 年度计数 → MPC 稳定性指标；就业报告指标 → 逐年决策 |
| 6 多智能体沙盘 | 300 名学生智能体 × 8 学期推演，3 专业 × 现状/方案 × 三情景的学生匹配度 |
| 7 ReplayBundle | `outputs/replay_bundle/` 中逐代轨迹、记忆库、Harness 基因组与 Qwen 调用原文的 SHA-256 清单与根哈希 |
| 9 2025 滚动复验 | 北京 10 所高校官网公布的 2025 年度增设专业（部分真值）上，EvoAlign 与惯性基线 Top-5 的命中与覆盖 |
| 8 预注册盲测 | 预注册文件哈希、登记早于数据抽取、冻结快照与脚本未改动；津冀/京外两个从未读取人群上的达标率、配对区间与 24 条假设判定 |

## 2. 关键结果的输入样例与输出

| 关键结果 | 最小输入样例 | 对应输出 | 复现方式 |
|---|---|---|---|
| 岗位→技能特征提取 | `data/verification/sample_jd_input.csv`（30 条脱敏岗位） | `sample_jd_output.csv`（岗位族 + 技能清单） | `python examples/extract_skills_demo.py` |
| 课程→技能抽取（Qwen） | `data/processed/university_curricula_enriched.json` 中的课程文本 | 同文件 `extracted_skills` 字段 | `outputs/replay_ledger.jsonl` 按请求哈希可查原始请求/响应 |
| 任务 1 三维差距 | `data/verification/benchmark_pathways.json` + `task1_targets.npz` | 《关键结果核对表》任务1基准 | 复核组 4 |
| 课程增删改与硬约束 | `outputs/plan_snapshots.json` 中 `B0_Freeze` 与 `Full_EvoAlign_OS` | 核对表「课程动作」、结果文件 `constraint_checks` | 复核组 1 |
| 达标率与配对区间 | `features_replay_2025.npz`（2,645 岗位 × 48 技能）+ 课程快照 | 核对表「六组消融」「配对区间」 | 复核组 1–2 |
| 招生调控 | `mpc_yearly_counts.csv`、`employment_feedback_series.json` | 核对表「就业反馈与招生」 | 复核组 5 |

## 3. 完整重跑（可选）

完整实验需要官方基础数据集清洗后的招聘切片（约 100 MB，超出提交体积限制，未随附），
由 `src/ingestion/build_demand_slices.py` 从官方数据生成，放在 `data/processed/` 与 `data/blind_holdout/` 后运行：

```bash
PYTHONPATH=src python src/experiments/run_unified_v2.py      # 约 7 分钟；大模型响应优先从重放账本读取
python src/experiments/generate_figures.py
PYTHONPATH=src python -m uvicorn server.cockpit_server:app --port 8765   # 在线驾驶舱
```

离线驾驶舱：直接打开 `web/index.html`（图表与全部结果可查看，实时重算需启动上述服务）。
调用大模型需在本机环境变量中提供自己的千问 API 密钥；本材料不含任何密钥。

## 4. 目录

| 路径 | 内容 |
|---|---|
| `自采数据登记表.xlsx` | 48 份自采公开文件的名称、网址、采集时间、方式、规模、合规结论与 SHA-256 |
| `关键结果核对表.xlsx` | 消融、配对区间、任务1、课程动作、风险矩阵、就业反馈、独立复核结果 |
| `数据合规自查声明.md` | 合规自查声明（待队长签字） |
| `outputs/` | 封印结果、课程快照、复核报告、大模型重放账本、隔离数据访问审计、图表 |
| `data/verification/` | 脱敏复核输入 |
| `data/processed/` | 知识图谱、培养方案课程、专业备案、紧缺目录、就业报告指标等结构化数据 |
| `src/` | 采集、本体、差距引擎、求解器、仿真、自进化、实验、复核与交付生成代码。正式实验入口为 `src/experiments/run_unified_v2.py`；`run_m3_experiments.py`、`run_structural_repair.py` 为早期版本，仅保留作对照，不产生报告数字 |
| `web/` | 决策驾驶舱 |
| `examples/` | 最小输入→输出演示脚本 |
"""


COMPLIANCE = """# 数据合规自查声明

本团队声明：方案所用数据均来自官方基础数据集或《赛题规则》允许的公开渠道；采集过程遵守来源网站使用协议、robots 协议与访问频率限制；
未采集含个人信息的数据；发布内容已按规定脱敏。

补充说明：
1. 自采数据仅为政府、高校与国际学术组织官网依法公开的文件，逐份登记于《自采数据登记表.xlsx》；
2. 招聘数据中的雇主名称以假名编号表示，职位描述中的联系方式与站点水印已删除；本材料仅含 30 条脱敏岗位样例与聚合特征矩阵；
3. 发送至大模型的内容仅为公开课程文本与聚合统计，不含个人信息；API 密钥未进入任何提交材料；
4. 工商数据按数据字典建议去除个体工商户，仅用于背景统计。

队长签字：＿＿＿＿＿＿　　日期：＿＿＿＿＿＿
"""

EXAMPLE = '''"""最小输入→输出演示：岗位文本 → 48 项标准技能 + 岗位族，并与 sample_jd_output.csv 逐行比对。"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from analytics.gap_engine import _compile_skill_regexes  # noqa: E402
from analytics.track_evaluator import classify_jd_track  # noqa: E402

kg = json.loads((ROOT / "data/processed/dynamic_kg.json").read_text(encoding="utf-8"))
regexes = _compile_skill_regexes(kg["skills"], None)
V = ROOT / "data/verification"
with (V / "sample_jd_output.csv").open(encoding="utf-8-sig") as f:
    expected = {r["job_id"]: r for r in csv.DictReader(f)}
ok = n = 0
with (V / "sample_jd_input.csv").open(encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        text = f"{r['job_title']} {r['description_redacted']}"
        hit = np.array([1 if pat.search(text) else 0 for _, pat in regexes], dtype=np.uint8)
        sids = [kg["skill_ids"][j] for j in np.nonzero(hit)[0]]
        track = classify_jd_track(r["job_title"], hit, kg["skill_ids"])
        e = expected[r["job_id"]]
        same = track == e["track"] and ";".join(sids) == e["skill_ids"]
        ok += same
        n += 1
        print(f"{r['job_title'][:16]:16s} -> {track:18s} {len(sids):2d} skills {'OK' if same else 'DIFF'}")
print(f"{ok}/{n} rows reproduce sample_jd_output.csv")
'''


def copy_tree(src: Path, dst: Path):
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", ".render-*"))


def scan_secrets(folder: Path):
    bad = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() not in (".png", ".mp4", ".npz", ".pdf", ".pptx", ".xlsx"):
            if SECRET.search(p.read_bytes()):
                bad.append(str(p))
        if p.name.startswith(".env"):
            bad.append(str(p))
    return bad


def dir_size(folder: Path) -> int:
    return sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", default=_TEAM.get("team_name") or "队伍名称")
    args = ap.parse_args()
    team = args.team
    m = json.loads((OUT / "verified_master_results.json").read_text(encoding="utf-8"))
    verify = json.loads((OUT / "verification_report.json").read_text(encoding="utf-8"))
    assert verify["summary"]["master_sha256"] == m["master_sha256"] and verify["summary"]["all_pass"], "run verify_key_results.py first"
    for need in ("方案报告.pdf", "答辩PPT.pptx", "演示视频.mp4", "演示脚本.md"):
        assert (DELIV / need).exists(), f"missing deliverable {need}"

    # Offline cockpit: embed the sealed master so web/index.html works from file:// without a server
    (ROOT / "web" / "master_embed.js").write_text(
        "window.__EMBEDDED_MASTER__ = " + json.dumps(m, ensure_ascii=False) + ";\n", encoding="utf-8")
    pkg_root = DELIV / "submission"
    if pkg_root.exists():
        shutil.rmtree(pkg_root)
    sup = pkg_root / f"{team}-数据支撑材料"
    (sup / "outputs").mkdir(parents=True)
    (sup / "data" / "processed").mkdir(parents=True)
    for n in OUT_KEEP:
        shutil.copy2(OUT / n, sup / "outputs" / n)
    copy_tree(OUT / "figures", sup / "outputs" / "figures")
    copy_tree(OUT / "replay_bundle", sup / "outputs" / "replay_bundle")
    for n in PROC_KEEP:
        shutil.copy2(PROC / n, sup / "data" / "processed" / n)
    copy_tree(VER, sup / "data" / "verification")
    shutil.copy2(ROOT / "data" / "self_collected_register.csv", sup / "data" / "self_collected_register.csv")
    copy_tree(ROOT / "src", sup / "src")
    copy_tree(ROOT / "web", sup / "web")
    (sup / "examples").mkdir()
    (sup / "examples" / "extract_skills_demo.py").write_text(EXAMPLE, encoding="utf-8")
    (sup / "requirements.txt").write_text(REQS, encoding="utf-8")
    (sup / "README.md").write_text(readme(m, verify, team), encoding="utf-8")
    (sup / "数据合规自查声明.md").write_text(COMPLIANCE, encoding="utf-8")
    (sup / "演示脚本.md").write_text((DELIV / "演示脚本.md").read_text(encoding="utf-8"), encoding="utf-8")
    register_xlsx(sup / "自采数据登记表.xlsx")
    results_xlsx(sup / "关键结果核对表.xlsx", m, verify)
    manifest = {str(p.relative_to(sup)): sha256(p) for p in sorted(sup.rglob("*")) if p.is_file()}
    (sup / "MANIFEST_SHA256.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")

    for sub, src, name in (("方案报告", "方案报告.pdf", "方案报告.pdf"), ("演示视频", "演示视频.mp4", "演示视频.mp4"), ("答辩PPT", "答辩PPT.pptx", "答辩PPT.pptx")):
        d = pkg_root / f"{team}-{sub}"
        d.mkdir()
        shutil.copy2(DELIV / src, d / f"{team}_{name}")

    bad = scan_secrets(pkg_root)
    if bad:
        raise SystemExit("secret-like content found: " + ", ".join(bad))
    limits = {f"{team}-数据支撑材料": 50e6, f"{team}-方案报告": 50e6, f"{team}-答辩PPT": 10e6}
    sizes = {p.name: dir_size(p) for p in pkg_root.iterdir()}
    for k, lim in limits.items():
        if sizes[k] > lim:
            raise SystemExit(f"{k} exceeds limit: {sizes[k] / 1e6:.1f} MB")
    zpath = DELIV / f"{team}.zip"
    if zpath.exists():
        zpath.unlink()
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in sorted(pkg_root.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(pkg_root).as_posix())
    with zipfile.ZipFile(DELIV / f"{team}-数据支撑材料.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in sorted(sup.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(pkg_root).as_posix())
    print(json.dumps({k: f"{v / 1e6:.2f} MB" for k, v in sizes.items()}, ensure_ascii=False))
    print(f"zip: {zpath} {zpath.stat().st_size / 1e6:.2f} MB; files in support package: {len(manifest)}")


if __name__ == "__main__":
    main()
