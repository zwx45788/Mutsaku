"""Refine M1 supply-side dataset:
1) Properly detect MOE 2023 & 2024 revocation sections (which use 撤销本科专业名单 headers).
2) Clean category classification across all Beijing universities (fixing sticky general_edu in BISTU/BTBU),
   parse THU_AI_IIIS text-based curriculum, and construct BJUT_CS_2024 from its 3 official syllabus volumes.
3) Output clean master tables in evoalign_os/data/processed/ and isolated natural-exp ground truths in blind_holdout/.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pdfplumber

RAW_DIR = Path("evoalign_os/data/raw_public")
TXT_DIR = RAW_DIR / "extracted_text"
PROC_DIR = Path("evoalign_os/data/processed")
BLIND_DIR = Path("evoalign_os/data/blind_holdout")

BEIJING_UNI_KEYWORDS = [
    "北京", "清华", "中国人民大学", "中央财经", "中央民族", "中央戏剧", "中央美术", "中央音乐",
    "中国农业大学", "中国政法大学", "中国石油大学", "中国矿业大学", "中国地质大学",
    "中国传媒大学", "中国青年政治", "中国劳动关系", "对外经济贸易大学", "华北电力大学",
    "北方工业大学", "首都经济贸易大学", "首都师范大学", "首都医科大学", "首都体育学院",
    "国际关系学院", "外交学院", "中国人民公安大学", "中国科学院大学", "中国社会科学院大学",
    "中华女子学院",
]

GEN_EDU_KW = (
    "大学英语", "英语", "体育", "马克思", "思想道德", "中国近现代史", "习近平", "毛泽东",
    "军事", "军训", "劳动教育", "国家安全", "形势与政策", "心理健康", "就业指导", "职业生涯",
    "大学语文", "国学", "老子", "大学写作", "传统文化", "哲学", "历史", "文学", "艺术",
    "俄语", "法语", "日语", "阿拉伯语", "报刊", "新媒体概论", "创新创业基础", "安全素养",
    "思政", "四史", "美育", "体育（", "大学体育",
)
MATH_SCI_KW = ("高等数学", "微积分", "线性代数", "概率论", "数理统计", "大学物理", "离散数学", "数值分析", "复变函数", "数学分析", "随机过程", "运筹学", "优化方法", "集合与图论", "代数结构")
PRACTICE_KW = ("课程设计", "实训", "实习", "毕业设计", "毕业论文", "综合实践", "创新实践", "工程实践", "实验（", "企业工作实践", "项目实践", "系统实践")
CORE_CS_KW = (
    "程序设计", "数据结构", "算法", "计算机组成", "操作系统", "计算机网络", "数据库",
    "编译原理", "软件工程", "计算机体系结构", "数字逻辑", "汇编", "计算概论", "计算机导论",
    "面向对象", "计算机系统", "人工智能导论", "机器学习", "深度学习", "模式识别",
)


def _clean(s: Any) -> str:
    return re.sub(r"\s+", "", str(s or ""))


def reparse_moe() -> None:
    all_recs: List[Dict[str, Any]] = []
    for year in (2021, 2022, 2023, 2024):
        pdf_path = RAW_DIR / f"MOE_{year}_APPROVALS.pdf"
        if not pdf_path.exists():
            continue
        txt_full = (TXT_DIR / f"MOE_{year}_APPROVALS.txt").read_text(encoding="utf-8", errors="replace")
        # Find exact page numbers where section titles appear in extracted text
        page_sections: Dict[int, str] = {}
        cur = "ADD_RECORD_MAJOR"
        pages_split = re.split(r"--- PAGE (\d+) ---", txt_full)
        for i in range(1, len(pages_split), 2):
            pno = int(pages_split[i])
            ptxt = pages_split[i + 1]
            # Check in order within the page text
            if "四、撤销专业名单" in ptxt or "撤销本科专业名单" in ptxt or "撤销专业名单" in ptxt:
                cur = "REVOKE_MAJOR"
            elif "三、调整学位授予门类或修业年限" in ptxt or "调整学位授予门类或修业年限专业名单" in ptxt:
                cur = "ADJUST_MAJOR"
            elif "二、新增审批本科专业名单" in ptxt:
                cur = "ADD_APPROVAL_MAJOR"
            elif "一、新增备案本科专业名单" in ptxt:
                cur = "ADD_RECORD_MAJOR"
            page_sections[pno] = cur

        with pdfplumber.open(str(pdf_path)) as pdf:
            for pno, page in enumerate(pdf.pages, start=1):
                act_page = page_sections.get(pno, "ADD_RECORD_MAJOR")
                # Also check if header changes mid-page
                ptxt = page.extract_text() or ""
                tables = page.extract_tables() or []
                for t_idx, tbl in enumerate(tables):
                    # If there are multiple tables on a page and "撤销" appears on page, check header cols
                    for row in tbl:
                        cells = [_clean(c) for c in row]
                        if len(cells) < 3:
                            continue
                        joined = " ".join(cells)
                        if "学校名称" in joined or "专业代码" in joined:
                            continue
                        code_idx = -1
                        for ci, c in enumerate(cells):
                            if re.fullmatch(r"\d{6}[A-Z]{0,2}", c):
                                code_idx = ci
                                break
                        if code_idx < 2:
                            continue
                        uni = cells[code_idx - 2]
                        major = cells[code_idx - 1]
                        code = cells[code_idx]
                        degree = cells[code_idx + 1] if code_idx + 1 < len(cells) else ""
                        dur = cells[code_idx + 2] if code_idx + 2 < len(cells) else ""
                        # Heuristic confirmation: revocation tables have no degree/duration columns
                        act = act_page
                        remark = cells[code_idx + 3] if code_idx + 3 < len(cells) else ""
                        if "调整学位" in remark or "调整修业" in remark or "调整学位" in joined:
                            act = "ADJUST_MAJOR"
                        elif "撤销" in remark:
                            act = "REVOKE_MAJOR"
                        elif act == "REVOKE_MAJOR" and degree in ("工学", "理学", "文学", "法学", "管理学", "经济学", "艺术学", "教育学", "医学", "农学", "历史学", "哲学") and dur in ("四年", "五年"):
                            act = "ADJUST_MAJOR"
                        all_recs.append({
                            "year": year,
                            "action": act,
                            "university": uni,
                            "is_beijing": any(k in uni for k in BEIJING_UNI_KEYWORDS),
                            "major_name": major,
                            "major_code": code,
                            "is_cs_ai_info": code.startswith(("0809", "0807", "0808", "0712", "1405")) or any(
                                k in major for k in ("计算机", "人工智能", "软件", "数据", "智能", "机器人", "网络", "信息", "电子", "集成电路", "量子", "算力")
                            ),
                            "degree": degree,
                            "duration": dur,
                            "page": pno,
                        })

    for yr in (2021, 2022, 2023, 2024):
        sub = [r for r in all_recs if r["year"] == yr]
        bj = [r for r in sub if r["is_beijing"]]
        print(
            f"MOE {yr}: nat={len(sub)}, bj={len(bj)} | "
            f"bj_add={sum(1 for r in bj if 'ADD' in r['action'])}, "
            f"bj_adj={sum(1 for r in bj if r['action']=='ADJUST_MAJOR')}, "
            f"bj_rev={sum(1 for r in bj if r['action']=='REVOKE_MAJOR')}"
        )

    moe_hist = [r for r in all_recs if r["year"] <= 2023]
    moe_2024 = [r for r in all_recs if r["year"] == 2024]
    (PROC_DIR / "moe_major_lifecycle_2021_2023.json").write_text(json.dumps(moe_hist, ensure_ascii=False, indent=2), encoding="utf-8")
    (BLIND_DIR / "moe_major_lifecycle_2024_blind.json").write_text(json.dumps(moe_2024, ensure_ascii=False, indent=2), encoding="utf-8")

    # Show 2021-2023 Beijing CS/AI revocations & additions
    bj_rev_cs = [(r["year"], r["university"], r["major_name"], r["major_code"]) for r in moe_hist if r["is_beijing"] and r["action"] == "REVOKE_MAJOR" and r["is_cs_ai_info"]]
    print("BJ 2021-2023 CS/AI/Info REVOKED:", bj_rev_cs)


def reclassify_course(c: Dict[str, Any]) -> Dict[str, Any]:
    name = c["course_name"]
    sec = c.get("section_raw", "")
    # Clean general_edu vs discipline_foundation vs major_core vs major_elective vs practice
    if any(k in name for k in GEN_EDU_KW):
        c["category"] = "general_edu"
        return c
    if any(k in name for k in MATH_SCI_KW):
        c["category"] = "discipline_foundation"
        c["required"] = True
        return c
    if any(k in name for k in PRACTICE_KW):
        c["category"] = "practice"
        return c
    if any(k in name for k in CORE_CS_KW):
        c["category"] = "major_core"
        c["required"] = True
        return c
    # For CUP_CS_2024, pages 3-6 were general education electives (humanties/history/etc)
    if c["program_id"] == "CUP_CS_2024" and c.get("source_page", 99) <= 6:
        c["category"] = "general_edu"
        c["required"] = False
        return c
    if "选修" in sec or "任选" in sec or "方向" in sec or not c.get("required", True):
        c["category"] = "major_elective"
        c["required"] = False
    else:
        c["category"] = "major_core"
    return c


def parse_thu_ai_iiis() -> List[Dict[str, Any]]:
    txt = (TXT_DIR / "U01_THU_AI_IIIS.txt").read_text(encoding="utf-8", errors="replace")
    courses = []
    seen = set()
    # Pattern: 8-digit code followed by course name and credit
    for m in re.finditer(r"(\d{8})\s+([^\n\d]{2,28}?)\s+(\d+(?:\.\d+)?)\s*(?:学分)?", txt):
        code, name, cr = m.group(1), m.group(2).strip(), float(m.group(3))
        if 0.5 <= cr <= 15 and name not in seen and "学分" not in name:
            seen.add(name)
            courses.append(reclassify_course({
                "program_id": "THU_AI_IIIS",
                "university": "清华大学",
                "tier": "tier1_research",
                "major": "计算机科学与技术(人工智能班)",
                "major_code": "080901",
                "cohort_year": 2024,
                "course_code": code,
                "course_name": name,
                "credits": cr,
                "total_hours": cr * 16.0,
                "lab_hours": 0.0,
                "practice_weeks": 0.0,
                "semester": 0,
                "category": "major_core",
                "required": True,
                "section_raw": "清华交叉院AI班",
                "source_page": 1,
            }))
    return courses


def build_bjut_program_from_syllabi(syllabi: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Construct BJUT CS curriculum table directly from its 3 official undergraduate syllabus volumes (610p)."""
    out = []
    seen = set()
    for s in syllabi:
        if s["university"] != "北京工业大学":
            continue
        name = s["course_name"]
        if name in seen:
            continue
        seen.add(name)
        cr = s.get("credits") or 3.0
        hr = s.get("total_hours") or cr * 16.0
        out.append(reclassify_course({
            "program_id": "BJUT_CS_SYLLABI",
            "university": "北京工业大学",
            "tier": "tier3_municipal",
            "major": "计算机科学与技术(大纲全集)",
            "major_code": "080901",
            "cohort_year": 2024,
            "course_code": "",
            "course_name": name,
            "credits": cr,
            "total_hours": hr,
            "lab_hours": round(hr * 0.25, 1) if s.get("has_lab_project") else 0.0,
            "practice_weeks": 0.0,
            "semester": 0,
            "category": "major_core",
            "required": True,
            "section_raw": "北工大本科教学大纲",
            "source_page": 1,
        }))
    return out


def main() -> None:
    reparse_moe()

    courses = json.loads((PROC_DIR / "university_curricula_courses.json").read_text(encoding="utf-8"))
    syllabi = json.loads((PROC_DIR / "course_syllabi_corpus.json").read_text(encoding="utf-8"))

    # Remove old empty THU_AI_IIIS if any and reclassify all courses
    courses = [reclassify_course(c) for c in courses if c["program_id"] != "THU_AI_IIIS"]
    courses.extend(parse_thu_ai_iiis())
    courses.extend(build_bjut_program_from_syllabi(syllabi))

    # Also add clean verified BISTU curriculum if parsed count looks inflated (due to duplicate table appendix)
    bistu_clean_csv = Path("产教智对_完整交付/data/references/curriculum_source.csv")
    if bistu_clean_csv.exists():
        import csv
        bistu_rows = []
        with open(bistu_clean_csv, "r", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                name = (r.get("course_name") or r.get("课程名称") or "").strip()
                if not name:
                    continue
                cr_m = re.search(r"(\d+(?:\.\d+)?)", str(r.get("credits") or r.get("学分") or "2.0"))
                cr = float(cr_m.group(1)) if cr_m else 2.0
                hr_raw = str(r.get("hours") or r.get("total_hours") or r.get("总学时") or "")
                pw = 0.0
                if "周" in hr_raw:
                    pw_m = re.search(r"(\d+(?:\.\d+)?)", hr_raw)
                    pw = float(pw_m.group(1)) if pw_m else cr
                    hr = 0.0
                else:
                    hr_m = re.search(r"(\d+(?:\.\d+)?)", hr_raw)
                    hr = float(hr_m.group(1)) if hr_m else cr * 16.0
                sem_raw = str(r.get("semester") or r.get("学期") or "0")
                sem_m = re.search(r"([1-8])", sem_raw)
                sem = int(sem_m.group(1)) if sem_m else 0
                req = "必" in str(r.get("required") or r.get("性质") or r.get("module") or "")
                bistu_rows.append(reclassify_course({
                    "program_id": "BISTU_CS_2025",
                    "university": "北京信息科技大学",
                    "tier": "tier3_municipal",
                    "major": "计算机科学与技术",
                    "major_code": "080901",
                    "cohort_year": 2025,
                    "course_code": str(r.get("course_code") or r.get("课程代码") or ""),
                    "course_name": name,
                    "credits": cr,
                    "total_hours": hr,
                    "lab_hours": float(r.get("lab_hours") or 0.0),
                    "practice_weeks": pw or float(r.get("practice_weeks") or 0.0),
                    "semester": sem,
                    "category": "major_core" if req else "major_elective",
                    "required": req,
                    "section_raw": str(r.get("module") or r.get("类别") or "BISTU核验表")[:35],
                    "source_page": 1,
                }))
        if len(bistu_rows) >= 40:
            courses = [c for c in courses if c["program_id"] != "BISTU_CS_2025"] + bistu_rows

    # Recompute clean summaries
    by_prog: Dict[str, List[Dict[str, Any]]] = {}
    for c in courses:
        by_prog.setdefault(c["program_id"], []).append(c)

    summaries = []
    for pid, cs in sorted(by_prog.items()):
        c0 = cs[0]
        tech = [x for x in cs if x["category"] != "general_edu"]
        summaries.append({
            "program_id": pid,
            "university": c0["university"],
            "tier": c0["tier"],
            "major": c0["major"],
            "major_code": c0["major_code"],
            "cohort_year": c0["cohort_year"],
            "total_courses": len(cs),
            "tech_courses": len(tech),
            "total_credits": round(sum(x["credits"] for x in cs), 1),
            "tech_credits": round(sum(x["credits"] for x in tech), 1),
        })

    # Recompute clean BTBU CS 2022 -> 2025 natural experiment truth (excluding general_edu)
    c22 = {c["course_name"] for c in by_prog.get("BTBU_CS_2022", []) if c["category"] != "general_edu"}
    c25 = {c["course_name"] for c in by_prog.get("BTBU_CS_2025", []) if c["category"] != "general_edu"}
    nat_truth = {
        "university": "北京工商大学",
        "major": "计算机科学与技术 (080901)",
        "from_year": 2022,
        "to_year": 2025,
        "tech_courses_2022": len(c22),
        "tech_courses_2025": len(c25),
        "added_in_2025": sorted(c25 - c22),
        "removed_from_2022": sorted(c22 - c25),
        "retained": sorted(c22 & c25),
    }

    (PROC_DIR / "university_curricula_courses.json").write_text(json.dumps(courses, ensure_ascii=False, indent=2), encoding="utf-8")
    (PROC_DIR / "university_programs_summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    (BLIND_DIR / "natural_exp_btbu_cs_2022_to_2025.json").write_text(json.dumps(nat_truth, ensure_ascii=False, indent=2), encoding="utf-8")

    unis = sorted({s["university"] for s in summaries})
    print(f"\nREFINED_CURRICULA: {len(summaries)} programs across {len(unis)} Beijing universities ({unis}), {len(courses)} courses")
    for s in summaries:
        print(f"  {s['program_id']:18s} | {s['university']:10s} | {s['tier']:18s} | {s['cohort_year']} | all={s['total_courses']:3d} tech={s['tech_courses']:3d} tech_cr={s['tech_credits']:5.1f}")
    print(f"CLEAN NAT_EXP BTBU CS (tech only) 2022->2025: +{len(nat_truth['added_in_2025'])} / -{len(nat_truth['removed_from_2022'])} / ={len(nat_truth['retained'])}")
    print("  Added tech courses in 2025:", nat_truth["added_in_2025"])
    print("  Removed tech courses from 2022:", nat_truth["removed_from_2022"])


if __name__ == "__main__":
    main()
