"""End-to-end audit of the submission against the competition's format rules and the single source of truth.

Run from the competition root after build_support_package.py:
    python evoalign_os/src/deliverables/final_audit.py --team 队伍名称
Writes outputs/deliverables/FINAL_AUDIT.json and prints a PASS/FAIL table.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import zipfile
from pathlib import Path

from PIL import Image
from pptx import Presentation

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
DELIV = OUT / "deliverables"
_TEAM_FILE = Path(__file__).resolve().parents[2] / "team_info.json"
_TEAM = json.loads(_TEAM_FILE.read_text(encoding="utf-8")) if _TEAM_FILE.exists() else {}
SECRET = re.compile(rb"sk-ws-[A-Za-z0-9._-]{8,}|sk-[A-Za-z0-9]{24,}")


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", default=_TEAM.get("team_name") or "队伍名称")
    team = ap.parse_args().team
    m = json.loads((OUT / "verified_master_results.json").read_text(encoding="utf-8"))
    sha = m["master_sha256"]
    master_file_sha = hashlib.sha256((OUT / "verified_master_results.json").read_bytes()).hexdigest()
    checks = []

    def chk(group, item, ok, detail=""):
        checks.append({"group": group, "item": item, "pass": bool(ok), "detail": str(detail)})

    zp = DELIV / f"{team}.zip"
    z = zipfile.ZipFile(zp)
    names = z.namelist()
    dirs = sorted({n.split("/")[0] for n in names})
    want = sorted(f"{team}-{x}" for x in ("方案报告", "数据支撑材料", "演示视频", "答辩PPT"))
    chk("打包", "ZIP 含四个规定目录", dirs == want, dirs)
    for d, f in (("方案报告", "方案报告.pdf"), ("演示视频", "演示视频.mp4"), ("答辩PPT", "答辩PPT.pptx")):
        chk("打包", f"命名 {team}_{f}", f"{team}-{d}/{team}_{f}" in names)
    sizes = {}
    for n in names:
        sizes[n.split("/")[0]] = sizes.get(n.split("/")[0], 0) + z.getinfo(n).file_size
    chk("打包", "数据支撑材料 ≤ 50MB", sizes[f"{team}-数据支撑材料"] <= 50e6, f"{sizes[f'{team}-数据支撑材料'] / 1e6:.1f} MB")
    leaks = [n for n in names if n.split("/")[-1].startswith(".env") or SECRET.search(z.read(n)) ]
    chk("合规", "提交包无 API 密钥/环境文件", not leaks, leaks[:3])

    pdf = DELIV / "方案报告.pdf"
    info = run(["pdfinfo", str(pdf)])
    pages = int(re.search(r"Pages:\s+(\d+)", info).group(1))
    chk("方案报告", "页数 ≤ 40", pages <= 40, pages)
    chk("方案报告", "体积 ≤ 50MB", pdf.stat().st_size <= 50e6, f"{pdf.stat().st_size / 1e6:.1f} MB")
    fonts = run(["pdffonts", str(pdf)]).splitlines()[2:]
    chk("方案报告", "字体全部嵌入", all(" yes " in f for f in fonts), f"{len(fonts)} fonts")
    text = run(["pdftotext", "-layout", str(pdf), "-"])
    for sec in ("现状差距分析", "方案设计", "效果验证与预测结果", "风险评估与应对", "数据说明", "合规自查声明"):
        chk("方案报告", f"含章节「{sec}」", sec in text)
    for k in ("赛题名称", "队伍名称", "所在高校", "队员名单及分工", "指导教师", "完成日期"):
        chk("方案报告", f"封面要素「{k}」", k in text)
    chk("方案报告", "封面信息已实际填写（无【待填写】/分工待填写）", "待填写" not in text.split("摘要")[0],
        {k: bool(_TEAM.get(k)) for k in ("team_name", "university", "advisor")} | {"roles": all(m.get("role") for m in _TEAM.get("members", []))})
    import stat as _stat, os as _os
    sec = Path(_os.path.expanduser("~/.config/evoalign_os/qwen.env"))
    mode = _stat.S_IMODE(sec.stat().st_mode) if sec.exists() else None
    chk("合规", "密钥仅存于本机 ext4 受保护文件（600/700），项目盘无密钥文件",
        mode == 0o600 and _stat.S_IMODE(sec.parent.stat().st_mode) == 0o700 and not any(ROOT.rglob(".env*")), oct(mode) if mode is not None else None)
    chk("单一事实源", "报告引用的封印哈希与结果文件一致", sha[:16] in text)
    c = m["cases"]["BTBU_CS_2022"]["results"]["replay_2025"]["metrics"]
    for v in (c["B0_Freeze"]["jd_qual_rate_70"], c["Full_EvoAlign_OS"]["jd_qual_rate_70"], m["task4_red_team_risks"]["governed_mean_breach_rate"]):
        chk("单一事实源", f"报告含结果值 {v * 100:.1f}%", f"{v * 100:.1f}%" in text)

    ppt = DELIV / "答辩PPT.pptx"
    chk("答辩PPT", "体积 ≤ 10MB", ppt.stat().st_size <= 10e6, f"{ppt.stat().st_size / 1e6:.1f} MB")
    prs = Presentation(str(ppt))
    titles = []
    for s in prs.slides:
        t = ""
        for sh in s.shapes:
            if sh.has_text_frame and sh.text_frame.text.strip():
                t = sh.text_frame.text.strip()
                break
        titles.append(t)
    order = [next((i for i, t in enumerate(titles) if t.startswith(k)), -1) for k in ("数据说明", "任务 1", "任务 2", "任务 3", "任务 4")]
    chk("答辩PPT", "按「数据说明 → 四项任务」顺序", -1 not in order and order == sorted(order), order)
    chk("单一事实源", "PPT 引用的封印哈希一致", any(sha[:12] in (sh.text_frame.text if sh.has_text_frame else "") for s in prs.slides for sh in s.shapes))

    mp4 = DELIV / "演示视频.mp4"
    dur = float(run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(mp4)]).strip())
    codecs = run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(mp4)]).split()
    chk("演示视频", "MP4 时长 ≤ 5 分钟", dur <= 300, f"{dur:.1f}s")
    chk("演示视频", "H.264 + AAC", codecs[:2] == ["h264", "aac"], codecs)
    script = (DELIV / "演示脚本.md").read_text(encoding="utf-8")
    chk("演示视频", "脚本覆盖建模主线", all(k in script for k in ("数据输入", "特征提取", "差距量化", "方案生成", "效果")))

    for fp in sorted((OUT / "figures").glob("*.png")):
        with Image.open(fp) as im:
            desc = json.loads(im.info.get("Description", "{}"))
        chk("图表溯源", f"{fp.name} 由当前结果文件生成", desc.get("verified_master_results.json") == master_file_sha)

    ver = json.loads((OUT / "verification_report.json").read_text(encoding="utf-8"))["summary"]
    chk("可复核", "独立复核全部一致", ver["all_pass"] and ver["master_sha256"] == sha, f"{ver['n_pass']}/{ver['n_checks']}")
    audit_lines = [json.loads(x) for x in (OUT / "blind_access_audit.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    callers = sorted({a["reason"].split(":")[0] for a in audit_lines})
    chk("防泄漏", "隔离数据仅经审计上下文读取", all(a.get("reason") for a in audit_lines), callers)
    for pid, case in m["cases"].items():
        cc = case["actions"]["Full_EvoAlign_OS"]["constraint_checks"]
        chk("硬约束", f"{pid} 推荐方案 {len(cc)} 项硬约束", all(cc.values()))
    sig = [case["results"][sp]["paired_qual_ci95"][ref]["Full_EvoAlign_OS"]["ci95_lo"] > 0
           for case in m["cases"].values() for sp in ("replay_2025", "replay_2026q1")
           for ref in ("B0_Freeze", "B1_SingleTrack_MILP", "B2_MultiTrack_Static", "B3_SingleShot_LLM")]
    chk("结论", "Full 相对 B0–B3 配对区间下界 > 0（两期 × 三专业）", all(sig), f"{sum(sig)}/{len(sig)}")

    import sys as _sys
    _sys.path.insert(0, str(ROOT / "src"))
    from rsi.replay_bundle import verify_replay_bundle
    vb = verify_replay_bundle(OUT / "replay_bundle")
    chk("自进化", "ReplayBundle 文件与根哈希完整、绑定当前结果", vb["files_ok"] and vb["root_ok"] and vb["master_sha256"] == sha)
    hist = m["rsi_full_evolution"]["evolution_history"]
    chk("自进化", "逐代轨迹含候选 Harness 基因组", any("candidate_harness" in h for h in hist))
    iis = m.get("task2_iis_diagnosis") or {}
    chk("求解器", "IIS 冲突诊断给出非空冲突集", iis.get("status") == "INFEASIBLE" and bool(iis.get("iis_conflicts")), iis.get("iis_conflicts", [])[:2])
    chk("任务3", "多智能体沙盘（3 专业 × 三情景）", len(m.get("task3_agent_dream_sim") or {}) == 3)
    cs = m.get("task1_cs101_core_coverage") or {}
    chk("数据", "101 计划核心课对照（11 份方案）", len(cs.get("programs", [])) == 11)
    esco = json.loads((ROOT / "data" / "processed" / "esco_alignment.json").read_text(encoding="utf-8"))["coverage"]
    chk("数据", "ESCO 已核实概念覆盖全部 48 项技能", esco["skills_with_verified_concept"] == 48, esco)
    hist_years = sorted({r["year"] for r in json.loads((ROOT / "data" / "processed" / m.get("moe_history_source", "")).read_text(encoding="utf-8"))})
    chk("数据", "教育部布点历史含 2020–2023 年度", hist_years[:1] == [2020] and hist_years[-1] == 2023, hist_years)
    pre_p, res_p = OUT / "preregistration_blind_v3.json", OUT / "preregistered_blind_results.json"
    if pre_p.exists() and res_p.exists():
        pre, bl = json.loads(pre_p.read_text(encoding="utf-8")), json.loads(res_p.read_text(encoding="utf-8"))
        body = {k: v for k, v in pre.items() if k != "preregistration_sha256"}
        ok_hash = hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest() == pre["preregistration_sha256"]
        first_extract = min(v["extracted_utc"] for v in bl["sealed_data"].values())
        chk("独立盲测", "预注册哈希完整且登记早于盲测数据抽取", ok_hash and pre["registered_utc"] < bl["run_started_utc"] <= first_extract,
            f"{pre['registered_utc']} < {first_extract}")
        chk("独立盲测", "执行脚本与登记版本一致", hashlib.sha256((ROOT / "src/experiments/preregistered_blind_test.py").read_bytes()).hexdigest() == pre["this_script_sha256"])
        chk("独立盲测", "盲测数据哈希与抽取记录一致", all(hashlib.sha256((ROOT / v["path"]).read_bytes()).hexdigest() == v["sha256"] for v in bl["sealed_data"].values()))
        chk("独立盲测", "冻结的封印结果未被改动", pre["frozen_master_sha256"] == sha, bl["hypotheses_supported"])
        chk("独立盲测", "报告包含预注册盲测章节", "预注册独立盲测" in text)
    mr_p = OUT / "macro_rolling_2025_partial.json"
    tr_p = ROOT / "data" / "blind_holdout" / "moe_major_lifecycle_2025_partial_univ.json"
    if mr_p.exists() and tr_p.exists():
        mr, tr = json.loads(mr_p.read_text(encoding="utf-8")), json.loads(tr_p.read_text(encoding="utf-8"))
        chk("数据", "2025 年度北京专业增撤：官网公告逐条核验并存档（部分替代）", tr["n_verified"] == tr["n_records"] and tr["n_records"] > 0 and
            hashlib.sha256(tr_p.read_bytes()).hexdigest() == mr["truth_sha256"], f"{tr['n_verified']}/{tr['n_records']} 条，{len(tr['universities'])} 所高校")
        chk("数据", "教育部布点历史 + 2025 滚动复验覆盖 2020–2025 年度", mr["history_years"] == [2020, 2021, 2022, 2023, 2024], mr["history_years"])
        chk("方案报告", "报告包含 2025 滚动复验", "滚动复验" in text)
    n_pass = sum(c["pass"] for c in checks)
    res = {"team": team, "master_sha256": sha, "n_checks": len(checks), "n_pass": n_pass, "all_pass": n_pass == len(checks), "checks": checks}
    (DELIV / "FINAL_AUDIT.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for c in checks:
        print(f"{'PASS' if c['pass'] else 'FAIL'}  [{c['group']}] {c['item']}  {c['detail']}")
    print(f"FINAL AUDIT {n_pass}/{len(checks)}")
    return 0 if res["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
