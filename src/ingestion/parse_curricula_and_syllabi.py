"""Structured parser for Beijing stratified university curricula & 610-page course syllabi.
Extracts structured course tables (code, name, credits, hours, lab_hours, semester, category, required)
across Tier 1 (PKU/THU/BUAA), Tier 2 (BJTU 4 majors, CUP, BJFU), Tier 3 (BTBU 2020-2025 CS/AI/SE/AUTO, BISTU 2025 CS),
plus real course-level syllabi from BJUT (610 pages) and BISTU (148 pages).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pdfplumber

RAW_DIR = Path("evoalign_os/data/raw_public")
TXT_DIR = RAW_DIR / "extracted_text"
PROC_DIR = Path("evoalign_os/data/processed")
BLIND_DIR = Path("evoalign_os/data/blind_holdout")
PROC_DIR.mkdir(parents=True, exist_ok=True)
BLIND_DIR.mkdir(parents=True, exist_ok=True)


def _clean(s: Any) -> str:
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s)).strip()


def _to_float(s: str) -> Optional[float]:
    s = _clean(s)
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _to_semester(s: str) -> int:
    s = _clean(s)
    # Match Chinese year/term like 大一/上, 大二/下
    cn_map = {
        "一上": 1, "一下": 2, "二上": 3, "二下": 4,
        "三上": 5, "三下": 6, "四上": 7, "四下": 8,
        "大一/上": 1, "大一/下": 2, "大二/上": 3, "大二/下": 4,
        "大三/上": 5, "大三/下": 6, "大四/上": 7, "大四/下": 8,
        "1秋": 1, "1春": 2, "2秋": 3, "2春": 4, "3秋": 5, "3春": 6, "4秋": 7, "4春": 8,
    }
    for k, v in cn_map.items():
        if k in s:
            return v
    m = re.search(r"([1-8])", s)
    if m:
        return int(m.group(1))
    return 0


def _infer_category(section_title: str, course_name: str, attr: str) -> Tuple[str, bool]:
    sec = section_title + " " + attr
    is_req = ("必修" in sec or "核心" in sec or "限选" in sec) and ("选修" not in attr and "任选" not in attr and "公选" not in attr)
    if any(k in sec for k in ("实践", "实习", "课程设计", "实训", "毕业设计", "毕业论文")) or any(
        k in course_name for k in ("课程设计", "实训", "实习", "毕业设计", "毕业论文", "综合实践")
    ):
        return "practice", True if "选" not in attr else False
    if any(k in sec for k in ("通识", "公共", "思政", "体育", "外语")) or any(
        k in course_name for k in ("大学英语", "体育", "马克思主义", "思想道德", "中国近现代史", "习近平", "毛泽东", "军事", "劳动教育", "国家安全")
    ):
        return "general_edu", is_req
    if any(k in sec for k in ("学科基础", "专业基础", "类定", "院定", "数理")):
        return "discipline_foundation", True if "选" not in attr else False
    if "选修" in sec or "任选" in sec or "拓展" in sec or "方向" in sec:
        return "major_elective", False
    return "major_core", is_req


def _is_valid_course_name(name: str) -> bool:
    if not name or len(name) < 2 or len(name) > 45:
        return False
    bad = [
        "课程名称", "学分小计", "总计", "合计", "备注", "说明", "模块", "修满", "不少于",
        "至少", "第一模块", "第二模块", "第三模块", "第四模块", "第五模块", "第六模块",
        "通识教育", "专业教育", "学科基础", "毕业要求", "指标点", "课程编码", "课程编号",
        "课程号", "小计", "学分要求", "必修课程", "选修课程",
    ]
    if any(b in name for b in bad):
        return False
    if re.fullmatch(r"[\d.\-/~、\s%]+", name):
        return False
    return True


def parse_pdf_tables_generic(
    pdf_path: Path,
    program_id: str,
    university: str,
    tier: str,
    major: str,
    major_code: str,
    cohort_year: int,
    page_range: Optional[Tuple[int, int]] = None,
) -> List[Dict[str, Any]]:
    courses: List[Dict[str, Any]] = []
    if not pdf_path.exists():
        return courses
    current_section = ""
    seen = set()

    with pdfplumber.open(str(pdf_path)) as pdf:
        p_start = (page_range[0] - 1) if page_range else 0
        p_end = min(page_range[1], len(pdf.pages)) if page_range else len(pdf.pages)
        for pno in range(p_start, p_end):
            page = pdf.pages[pno]
            page_txt = page.extract_text() or ""
            # Skip graduation requirement matrix pages that only have H/M/L or 0.1/0.2 weights
            if "支撑强度" in page_txt or "指标点分解" in page_txt or "毕业要求实现矩阵" in page_txt:
                if "课程编码" not in page_txt and "课程编号" not in page_txt and "课程代码" not in page_txt and "总学时" not in page_txt:
                    continue
            for line in page_txt.splitlines():
                ls = line.strip()
                if any(k in ls for k in ("通识基础", "通识选修", "学科基础", "专业核心", "专业必修", "专业选修", "集中实践", "实践教学", "专业方向", "类定课程", "院定课程")):
                    if len(ls) < 35:
                        current_section = ls

            tables = page.extract_tables() or []
            for tbl in tables:
                if not tbl or len(tbl) < 2:
                    continue
                # Find header row within first 3 rows
                hdr_idx = -1
                col_map: Dict[str, int] = {}
                for r_i, row in enumerate(tbl[:4]):
                    cells = [_clean(c) for c in row]
                    joined = " ".join(cells)
                    if ("课程名称" in joined or "课程名" in joined) and ("学分" in joined or "学时" in joined):
                        hdr_idx = r_i
                        for c_i, cell in enumerate(cells):
                            if "课程编码" in cell or "课程编号" in cell or "课程代码" in cell or "课程号" in cell:
                                col_map.setdefault("code", c_i)
                            elif "课程名称" in cell or "课程名" in cell:
                                col_map.setdefault("name", c_i)
                            elif "学分" in cell and "占" not in cell:
                                col_map.setdefault("credits", c_i)
                            elif "总学时" in cell or ("学时" in cell and "实验" not in cell and "实践" not in cell and "周" not in cell and "理论" not in cell and "上机" not in cell):
                                col_map.setdefault("hours", c_i)
                            elif "实验" in cell or "实践" in cell or "上机" in cell:
                                col_map.setdefault("lab_hours", c_i)
                            elif "学期" in cell:
                                col_map.setdefault("semester", c_i)
                            elif "属性" in cell or "性质" in cell or "类别" in cell:
                                col_map.setdefault("attr", c_i)
                        break

                # Sometimes continuation tables omit header row; infer from previous layout if columns look like code/name/credits/hours
                start_row = hdr_idx + 1 if hdr_idx >= 0 else 0
                if hdr_idx == -1:
                    # Try heuristic: check if rows have course code pattern + Chinese title + numeric credit
                    sample_ok = 0
                    for row in tbl[:6]:
                        cells = [_clean(c) for c in row]
                        if len(cells) >= 4:
                            if re.match(r"^[A-Za-z0-9]{5,14}$", cells[0]) and _to_float(cells[2]) is not None:
                                sample_ok += 1
                            elif len(cells) >= 5 and re.match(r"^[A-Za-z0-9]{5,14}$", cells[1]) and _to_float(cells[3]) is not None:
                                sample_ok += 1
                    if sample_ok >= 2:
                        first_cells = [_clean(c) for c in tbl[0]]
                        if re.match(r"^[A-Za-z0-9]{5,14}$", first_cells[0]):
                            col_map = {"code": 0, "name": 1, "credits": 2, "hours": 3, "lab_hours": 4 if len(first_cells) > 4 else -1, "semester": 5 if len(first_cells) > 5 else -1, "attr": 6 if len(first_cells) > 6 else -1}
                        else:
                            col_map = {"code": 1, "name": 2, "credits": 3, "hours": 4, "lab_hours": 5 if len(first_cells) > 5 else -1, "semester": 6 if len(first_cells) > 6 else -1, "attr": 7 if len(first_cells) > 7 else -1}
                    else:
                        continue

                if "name" not in col_map or "credits" not in col_map:
                    continue

                for row in tbl[start_row:]:
                    cells = [_clean(c) for c in row]
                    if not any(cells):
                        continue
                    row_joined = " ".join(cells)
                    if any(k in row_joined for k in ("通识基础", "通识选修", "学科基础", "专业核心", "专业必修", "专业选修", "集中实践", "实践教学", "方向模块")):
                        if len(row_joined) < 40:
                            current_section = row_joined
                            continue

                    def get_c(key: str) -> str:
                        idx = col_map.get(key, -1)
                        return cells[idx] if 0 <= idx < len(cells) else ""

                    code = get_c("code")
                    name = get_c("name")
                    # Sometimes code and name are merged in cell 0 like "COMA2G1001 人工智能导论A"
                    if not name and code and " " in code:
                        parts = code.split(" ", 1)
                        code, name = parts[0], parts[1]
                    elif name and re.match(r"^[A-Za-z0-9]{6,14}\s+", name):
                        m_sp = re.match(r"^([A-Za-z0-9]{6,14})\s+(.*)$", name)
                        if m_sp:
                            code = code or m_sp.group(1)
                            name = m_sp.group(2)

                    if not _is_valid_course_name(name):
                        continue
                    credits = _to_float(get_c("credits"))
                    if credits is None or credits <= 0 or credits > 20:
                        continue
                    hours = _to_float(get_c("hours"))
                    if hours is None:
                        hours = credits * 16.0
                    # Handle week-based practice hours like "2周"
                    raw_h = get_c("hours")
                    practice_weeks = 0.0
                    if "周" in raw_h:
                        pw = _to_float(raw_h)
                        practice_weeks = pw or credits
                        hours = 0.0
                    lab_hours = _to_float(get_c("lab_hours")) or 0.0
                    if lab_hours > hours and hours > 0:
                        lab_hours = 0.0
                    sem = _to_semester(get_c("semester"))
                    attr = get_c("attr")
                    cat, is_req = _infer_category(current_section, name, attr)
                    if cat == "practice" and practice_weeks == 0.0 and hours <= 4:
                        practice_weeks = credits

                    dedup_key = (program_id, re.sub(r"\s+", "", name), round(credits, 1))
                    if dedup_key in seen:
                        continue
                    seen.add(dedup_key)

                    courses.append({
                        "program_id": program_id,
                        "university": university,
                        "tier": tier,
                        "major": major,
                        "major_code": major_code,
                        "cohort_year": cohort_year,
                        "course_code": code,
                        "course_name": name,
                        "credits": round(credits, 2),
                        "total_hours": round(hours, 1),
                        "lab_hours": round(lab_hours, 1),
                        "practice_weeks": round(practice_weeks, 1),
                        "semester": sem,
                        "category": cat,
                        "required": is_req,
                        "section_raw": current_section[:40],
                        "source_page": pno + 1,
                    })
    return courses


def parse_buaa_html() -> List[Dict[str, Any]]:
    """Parse BUAA 2025 CS curriculum from extracted text."""
    p = TXT_DIR / "U03_BUAA_CS_2025.txt"
    if not p.exists():
        return []
    text = p.read_text(encoding="utf-8", errors="replace")
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    courses = []
    seen = set()
    # BUAA course codes look like B3I061010 or B1A09104A
    for i, ln in enumerate(lines):
        if re.fullmatch(r"[A-Z][0-9A-Z]{6,11}", ln) and i + 3 < len(lines):
            name = lines[i + 1]
            if not _is_valid_course_name(name):
                continue
            # Look ahead up to 5 lines for credit and hours
            cr = None
            hr = None
            for k in range(2, 6):
                if i + k < len(lines):
                    v = _to_float(lines[i + k])
                    if v is not None and cr is None and 0.5 <= v <= 16:
                        cr = v
                    elif v is not None and cr is not None and hr is None and v >= 8:
                        hr = v
                        break
            if cr is not None:
                key = ("U03_BUAA_CS_2025", name, round(cr, 1))
                if key not in seen:
                    seen.add(key)
                    cat, is_req = _infer_category("", name, "必修")
                    courses.append({
                        "program_id": "U03_BUAA_CS_2025",
                        "university": "北京航空航天大学",
                        "tier": "tier1_research",
                        "major": "计算机科学与技术",
                        "major_code": "080901",
                        "cohort_year": 2025,
                        "course_code": ln,
                        "course_name": name,
                        "credits": cr,
                        "total_hours": hr or cr * 16.0,
                        "lab_hours": 0.0,
                        "practice_weeks": 0.0,
                        "semester": 0,
                        "category": cat,
                        "required": is_req,
                        "section_raw": "北航培养方案",
                        "source_page": 1,
                    })
    return courses


def parse_bjut_and_bistu_syllabi() -> List[Dict[str, Any]]:
    """Extract course-level full syllabi from BJUT (610p) and BISTU (148p) extracted texts."""
    syllabi: List[Dict[str, Any]] = []
    for fid, uni in [
        ("U12_BJUT_SYLLABUS_1", "北京工业大学"),
        ("U12_BJUT_SYLLABUS_2", "北京工业大学"),
        ("U12_BJUT_SYLLABUS_3", "北京工业大学"),
        ("U13_BISTU_SYLLABUS", "北京信息科技大学"),
    ]:
        p = TXT_DIR / f"{fid}.txt"
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        # Split on syllabus headers like “课程名称”课程教学大纲 or 《...》教学大纲
        parts = re.split(r"(?=(?:[“《][^”》\n]{2,35}[”》]\s*课程?教学大纲|课程名称[：:]\s*\S+))", text)
        for seg in parts:
            seg_s = seg.strip()
            if len(seg_s) < 300:
                continue
            m_title = re.search(r"[“《]([^”》\n]{2,35})[”》]\s*课程?教学大纲", seg_s[:160])
            if not m_title:
                m_title = re.search(r"课程名称[：:]\s*([^\n\r]{2,35})", seg_s[:220])
            if not m_title:
                continue
            cname = _clean(m_title.group(1))
            if not _is_valid_course_name(cname):
                continue
            # Extract objectives & content summary (bounded cleanly)
            obj_m = re.search(r"(?:课程目标|教学目标)([\s\S]{30,800})", seg_s)
            con_m = re.search(r"(?:教学内容|课程内容|主要内容)([\s\S]{30,1200})", seg_s)
            # Extract lab/practice indicators
            has_lab = any(k in seg_s for k in ("实验", "上机", "大作业", "项目", "编程实现", "系统设计"))
            credits_m = re.search(r"学\s*分[：:\s]*(\d+(?:\.\d+)?)", seg_s[:500])
            hours_m = re.search(r"(?:总学时|学\s*时)[：:\s]*(\d+(?:\.\d+)?)", seg_s[:500])
            syllabi.append({
                "source_id": fid,
                "university": uni,
                "course_name": cname,
                "credits": float(credits_m.group(1)) if credits_m else None,
                "total_hours": float(hours_m.group(1)) if hours_m else None,
                "has_lab_project": has_lab,
                "objectives_snippet": _clean(obj_m.group(1))[:500] if obj_m else "",
                "content_snippet": _clean(con_m.group(1))[:800] if con_m else _clean(seg_s)[:800],
                "full_chars": len(seg_s),
            })

    # Deduplicate syllabi by (university, normalized course_name), keeping richest text
    best: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for s in syllabi:
        key = (s["university"], re.sub(r"[ⅠⅡⅢ123ABC\(\)（）\s]", "", s["course_name"]))
        if key not in best or s["full_chars"] > best[key]["full_chars"]:
            best[key] = s
    return list(best.values())


def main() -> None:
    targets = [
        # Tier 3 Primary Portfolio: BTBU CS + AI + SE + AUTO (2020-2025!)
        ("U11_BTBU_CS_2020.pdf", "BTBU_CS_2020", "北京工商大学", "tier3_municipal", "计算机科学与技术", "080901", 2020, None),
        ("U11_BTBU_CS_2022.pdf", "BTBU_CS_2022", "北京工商大学", "tier3_municipal", "计算机科学与技术", "080901", 2022, None),
        ("U11_BTBU_CS_2023.pdf", "BTBU_CS_2023", "北京工商大学", "tier3_municipal", "计算机科学与技术", "080901", 2023, None),
        ("U11_BTBU_CS_2025.pdf", "BTBU_CS_2025", "北京工商大学", "tier3_municipal", "计算机科学与技术", "080901", 2025, None),
        ("U11_BTBU_AI_2021.pdf", "BTBU_AI_2021", "北京工商大学", "tier3_municipal", "人工智能", "080717T", 2021, None),
        ("U11_BTBU_AI_2022.pdf", "BTBU_AI_2022", "北京工商大学", "tier3_municipal", "人工智能", "080717T", 2022, None),
        ("U11_BTBU_AI_2025.pdf", "BTBU_AI_2025", "北京工商大学", "tier3_municipal", "人工智能", "080717T", 2025, None),
        ("U11_BTBU_SE_2022.pdf", "BTBU_SE_2022", "北京工商大学", "tier3_municipal", "软件工程", "080902", 2022, None),
        ("U11_BTBU_SE_2025.pdf", "BTBU_SE_2025", "北京工商大学", "tier3_municipal", "软件工程", "080902", 2025, None),
        ("U11_BTBU_AUTO_2024.pdf", "BTBU_AUTO_2024", "北京工商大学", "tier3_municipal", "自动化", "080801", 2024, None),
        # Tier 3 Cross-School Transfer: BISTU CS 2025
        ("U13_BISTU_CS_2025.pdf", "BISTU_CS_2025", "北京信息科技大学", "tier3_municipal", "计算机科学与技术", "080901", 2025, None),
        # Tier 2 Industry Double-First-Class: BJTU 4-major college portfolio + CUP + BJFU
        ("U06_BJTU_CS_2026.pdf", "BJTU_CS_2025", "北京交通大学", "tier2_double_first", "计算机科学与技术", "080901", 2025, (6, 30)),
        ("U06_BJTU_CS_2026.pdf", "BJTU_AI_2025", "北京交通大学", "tier2_double_first", "人工智能", "080717T", 2025, (47, 68)),
        ("U06_BJTU_CS_2026.pdf", "BJTU_IOT_2025", "北京交通大学", "tier2_double_first", "物联网工程", "080905", 2025, (69, 88)),
        ("U06_BJTU_CS_2026.pdf", "BJTU_DS_2025", "北京交通大学", "tier2_double_first", "数据科学与大数据技术", "080910T", 2025, (89, 105)),
        ("U07_CUP_CS_2024.pdf", "CUP_CS_2024", "中国石油大学(北京)", "tier2_double_first", "计算机科学与技术", "080901", 2024, None),
        ("U08_BJFU_INFO_2022.pdf", "BJFU_CS_2022", "北京林业大学", "tier2_double_first", "计算机科学与技术", "080901", 2022, None),
        # Tier 1 Frontier Research Benchmarks: PKU + THU
        ("U02_PKU_CS_2024.pdf", "PKU_CS_2024", "北京大学", "tier1_research", "计算机科学与技术", "080901", 2024, None),
        ("U02_PKU_TURING_2025.pdf", "PKU_TURING_2025", "北京大学", "tier1_research", "计算机科学与技术(图灵班)", "080901", 2025, None),
        ("U01_THU_AI_IIIS.pdf", "THU_AI_IIIS", "清华大学", "tier1_research", "计算机科学与技术(人工智能班)", "080901", 2024, None),
    ]

    all_courses: List[Dict[str, Any]] = []
    program_summaries: List[Dict[str, Any]] = []

    for fname, pid, uni, tier, major, mcode, yr, prange in targets:
        cs = parse_pdf_tables_generic(RAW_DIR / fname, pid, uni, tier, major, mcode, yr, prange)
        all_courses.extend(cs)
        tech_cs = [c for c in cs if c["category"] != "general_edu"]
        program_summaries.append({
            "program_id": pid,
            "university": uni,
            "tier": tier,
            "major": major,
            "major_code": mcode,
            "cohort_year": yr,
            "total_courses": len(cs),
            "tech_courses": len(tech_cs),
            "total_credits_parsed": round(sum(c["credits"] for c in cs), 1),
            "tech_credits_parsed": round(sum(c["credits"] for c in tech_cs), 1),
        })

    # Add BUAA HTML
    buaa_cs = parse_buaa_html()
    all_courses.extend(buaa_cs)
    program_summaries.append({
        "program_id": "U03_BUAA_CS_2025",
        "university": "北京航空航天大学",
        "tier": "tier1_research",
        "major": "计算机科学与技术",
        "major_code": "080901",
        "cohort_year": 2025,
        "total_courses": len(buaa_cs),
        "tech_courses": sum(1 for c in buaa_cs if c["category"] != "general_edu"),
        "total_credits_parsed": round(sum(c["credits"] for c in buaa_cs), 1),
        "tech_credits_parsed": round(sum(c["credits"] for c in buaa_cs if c["category"] != "general_edu"), 1),
    })

    # Parse full syllabi (BJUT 610p + BISTU 148p)
    syllabi = parse_bjut_and_bistu_syllabi()

    # Physically isolate BTBU_CS_2025 / BTBU_AI_2025 / BTBU_SE_2025 course-level diff into blind_holdout for natural experiment
    btbu_2022_cs = {c["course_name"] for c in all_courses if c["program_id"] == "BTBU_CS_2022" and c["category"] != "general_edu"}
    btbu_2025_cs = {c["course_name"] for c in all_courses if c["program_id"] == "BTBU_CS_2025" and c["category"] != "general_edu"}
    nat_exp_course_truth = {
        "university": "北京工商大学",
        "major": "计算机科学与技术 (080901)",
        "from_year": 2022,
        "to_year": 2025,
        "courses_2022_count": len(btbu_2022_cs),
        "courses_2025_count": len(btbu_2025_cs),
        "added_in_2025": sorted(btbu_2025_cs - btbu_2022_cs),
        "removed_from_2022": sorted(btbu_2022_cs - btbu_2025_cs),
        "retained": sorted(btbu_2022_cs & btbu_2025_cs),
    }

    (PROC_DIR / "university_curricula_courses.json").write_text(
        json.dumps(all_courses, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (PROC_DIR / "university_programs_summary.json").write_text(
        json.dumps(program_summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (PROC_DIR / "course_syllabi_corpus.json").write_text(
        json.dumps(syllabi, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (BLIND_DIR / "natural_exp_btbu_cs_2022_to_2025.json").write_text(
        json.dumps(nat_exp_course_truth, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"CURRICULA_PARSED: {len(program_summaries)} programs across {len({p['university'] for p in program_summaries})} Beijing universities, {len(all_courses)} total courses, {len(syllabi)} full syllabi")
    for s in program_summaries:
        print(f"  {s['program_id']:18s} | {s['university']:10s} | {s['major']:14s} | {s['cohort_year']} | all={s['total_courses']:3d} tech={s['tech_courses']:3d} cr={s['total_credits_parsed']}")
    print(f"NAT_EXP_COURSE_TRUTH (BTBU CS 2022->2025): +{len(nat_exp_course_truth['added_in_2025'])} added, -{len(nat_exp_course_truth['removed_from_2022'])} removed, ={len(nat_exp_course_truth['retained'])} retained")
    print("  Sample added in 2025:", nat_exp_course_truth["added_in_2025"][:15])


if __name__ == "__main__":
    main()
