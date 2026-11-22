"""Ingest additional official sources requested by the M1 data contract.

1. MOE 2020年度普通高校本科专业备案和审批结果 (教高函〔2021〕1号, XLS) -> merged with 2021-2023 into
   data/processed/moe_major_lifecycle_2020_2023.json (2024 stays in blind_holdout/).
2. MOE 《普通高等学校本科专业目录（2026年）》(教高函〔2026〕2号, PDF) -> data/processed/moe_undergrad_catalog_2026.json
3. 计算机领域“101计划”12 门核心课 (国家高等教育智慧教育平台) -> data/processed/cs101_core_courses.json
Each source gets a provenance row (URL, UTC time, bytes, SHA-256) in data/self_collected_register.csv.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ingestion.parse_moe_and_bj_catalog import _is_beijing_uni

RAW = Path("evoalign_os/data/raw_public")
PROC = Path("evoalign_os/data/processed")
REG = Path("evoalign_os/data/self_collected_register.csv")
NOW = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
CS_CODES = ("0809", "0807", "0808", "0712", "1405", "0810")
CS_WORDS = ("计算机", "人工智能", "软件", "数据", "智能", "机器人", "网络", "信息", "电子", "集成电路", "算力", "量子")
SOURCES = {
    "MOE_2020_APPROVALS": ("moe_major_lifecycle", "教育部", "2020年度本科专业备案审批与撤销结果（教高函〔2021〕1号）", "2020",
                           "https://www.moe.gov.cn/srcsite/A08/moe_1034/s4930/202103/W020210302545152199812.xls", RAW / "MOE_2020_APPROVALS.xls"),
    "MOE_2021_NEW_MAJOR_LIST": ("moe_major_lifecycle", "教育部", "列入本科专业目录的新专业名单（2021年）", "2021",
                                "https://www.moe.gov.cn/srcsite/A08/moe_1034/s4930/202103/W020210301421098802148.xls", RAW / "MOE_2021_NEW_MAJOR_LIST.xls"),
    "MOE_CATALOG_2026": ("moe_major_catalog", "教育部", "普通高等学校本科专业目录（2026年，教高函〔2026〕2号）", "2026",
                         "https://www.moe.gov.cn/srcsite/A08/moe_1034/s3882/202604/W020260427440749576927.pdf", RAW / "MOE_CATALOG_2026.pdf"),
}
CS101_URL = "https://higher.smartedu.cn/project101/%E8%AE%A1%E7%AE%97%E6%9C%BA%E7%A7%91%E5%AD%A6"
CS101 = [
    ("计算概论", "哈尔滨工业大学", ["c_cpp", "python_eng"], ["计算概论", "程序设计基础", "高级语言程序设计", "C语言", "计算机导论", "Python程序设计"]),
    ("数据结构", "上海交通大学", ["dsa"], ["数据结构"]),
    ("算法设计与分析", "北京大学", ["dsa"], ["算法设计", "算法分析"]),
    ("离散数学", "北京大学", ["discrete_math"], ["离散数学", "离散结构"]),
    ("计算机系统导论", "南京大学", ["comp_arch", "os_kernel", "c_cpp"], ["计算机系统导论", "计算机系统基础", "计算机系统原理"]),
    ("操作系统", "北京大学", ["os_kernel"], ["操作系统"]),
    ("计算机组成与系统结构", "清华大学", ["comp_arch"], ["组成原理", "计算机组成", "体系结构", "系统结构"]),
    ("编译原理", "北京航空航天大学", ["compiler_pl"], ["编译"]),
    ("计算机网络", "清华大学", ["comp_net"], ["计算机网络", "网络原理"]),
    ("数据库系统", "中国人民大学", ["sql_rdbms"], ["数据库"]),
    ("软件工程", "国防科技大学", ["soft_eng_arch"], ["软件工程"]),
    ("人工智能引论", "浙江大学", ["ml_classic", "deep_learning"], ["人工智能引论", "人工智能导论", "人工智能基础", "人工智能原理"]),
]
CS101_SCHOOLS = ("北京大学、清华大学、北京航空航天大学、北京理工大学、哈尔滨工业大学、上海交通大学、南京大学、浙江大学、华中科技大学、电子科技大学、"
                 "西安交通大学、国防科技大学、北京邮电大学、中国科学院大学、吉林大学、同济大学、中国科学技术大学、武汉大学、中南大学、西北工业大学、"
                 "西安电子科技大学、中国人民大学、北京交通大学、天津大学、大连理工大学、复旦大学、华东师范大学、东南大学、山东大学、湖南大学、"
                 "中山大学、华南理工大学、重庆大学").split("、")


def _clean(x) -> str:
    return "" if x is None or (isinstance(x, float) and pd.isna(x)) else re.sub(r"\s+", "", str(x))


def parse_2020() -> list:
    df = pd.read_excel(SOURCES["MOE_2020_APPROVALS"][5], sheet_name=0, header=None, dtype=str)
    action, recs = None, []
    for _, row in df.iterrows():
        cells = [_clean(c) for c in row.tolist()]
        head = cells[0]
        if re.match(r"^[一二三四五六]、", head):
            action = ("ADD_RECORD_MAJOR" if "备案" in head and "新增" in head else
                      "ADD_APPROVAL_MAJOR" if "审批" in head and "新增" in head else
                      "REVOKE_MAJOR" if "撤销" in head else "ADJUST_MAJOR")
            continue
        code_i = next((i for i, c in enumerate(cells) if re.fullmatch(r"\d{6}[A-Z]{0,2}", c)), None)
        if action is None or code_i is None or code_i < 2 or not cells[0].isdigit():
            continue
        uni, major, code = cells[code_i - 2], cells[code_i - 1], cells[code_i]
        recs.append({"year": 2020, "action": action, "university": uni, "is_beijing": _is_beijing_uni(uni),
                     "major_name": major, "major_code": code,
                     "is_cs_ai_info": code.startswith(CS_CODES) or any(k in major for k in CS_WORDS),
                     "degree": cells[code_i + 1] if code_i + 1 < len(cells) else "",
                     "duration": cells[code_i + 2] if code_i + 2 < len(cells) else "", "page": None, "source": "MOE_2020_APPROVALS.xls"})
    return recs


def parse_catalog() -> list:
    txt = subprocess.run(["pdftotext", "-layout", str(SOURCES["MOE_CATALOG_2026"][5]), "-"], capture_output=True, text=True, check=True).stdout
    out, cls = [], None
    for ln in txt.splitlines():
        m = re.match(r"^\s*(\d{4})\s{2,}(\S.*?)\s*$", ln)
        if m:
            cls = (m.group(1), m.group(2))
            continue
        m = re.match(r"^\s*(\d{6}[A-Z]{0,2})\s+(\S+?)(?:（注：(.*))?\s*$", ln)
        if m:
            out.append({"code": m.group(1), "name": m.group(2), "class_code": cls[0] if cls else m.group(1)[:4],
                        "class_name": cls[1] if cls else "", "note": (m.group(3) or "").rstrip("）")})
    return out


def register(rows_new):
    rows = [r for r in csv.reader(REG.open(encoding="utf-8")) if r and r[0] not in {x[0] for x in rows_new}]
    rows += rows_new
    with REG.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)


def main():
    r2020 = parse_2020()
    old = json.loads((PROC / "moe_major_lifecycle_2021_2023.json").read_text(encoding="utf-8"))
    hist = r2020 + old
    (PROC / "moe_major_lifecycle_2020_2023.json").write_text(json.dumps(hist, ensure_ascii=False), encoding="utf-8")
    cat = parse_catalog()
    (PROC / "moe_undergrad_catalog_2026.json").write_text(json.dumps({"source": SOURCES["MOE_CATALOG_2026"][4], "n_majors": len(cat), "majors": cat}, ensure_ascii=False, indent=1), encoding="utf-8")
    spec = {"source": "国家高等教育智慧教育平台·计算机领域本科教育教学改革试点工作计划（101计划）计算机科学核心课程",
            "url": CS101_URL, "retrieved_utc": NOW, "pilot_universities": CS101_SCHOOLS,
            "pilot_universities_source": "高等教育出版社《高等学校计算机类专业人才培养战略研究报告暨核心课程体系》(2023) 目录",
            "skill_mapping_note": "课程→技能映射由团队按课程名称与 CS2023 知识领域人工指定",
            "courses": [{"name": n, "lead_university": u, "skills": s, "name_keywords": k} for n, u, s, k in CS101]}
    (PROC / "cs101_core_courses.json").write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
    rows = []
    for sid, (cat_, org, name, yr, url, path) in SOURCES.items():
        raw = path.read_bytes()
        rows.append([sid, cat_, org, name, yr, url, NOW, "ok", str(len(raw)), "", "", hashlib.sha256(raw).hexdigest(),
                     "教育部官网依法公开附件；仅提取高校-专业-代码结构化字段，不含个人信息"])
    sp = (PROC / "cs101_core_courses.json").read_bytes()
    rows.append(["STD_CS101_PLAN", "standard_ontology", "教育部高教司/国家高等教育智慧教育平台", "计算机领域101计划12门核心课与33所试点高校", "2021-2023",
                 CS101_URL, NOW, "ok", str(len(sp)), "", "", hashlib.sha256(sp).hexdigest(), "官方平台公开页面；仅录入课程名称与牵头高校"])
    register(rows)
    bj = [r for r in r2020 if r["is_beijing"]]
    print(f"2020: {len(r2020)} records ({len(bj)} Beijing, {sum(1 for r in bj if r['is_cs_ai_info'])} Beijing CS/AI/info);",
          {a: sum(1 for r in r2020 if r['action'] == a) for a in ('ADD_RECORD_MAJOR', 'ADD_APPROVAL_MAJOR', 'ADJUST_MAJOR', 'REVOKE_MAJOR')})
    print(f"history 2020-2023: {len(hist)} records; catalog 2026: {len(cat)} majors; 101 courses: {len(CS101)}")
    print("sample BJ 2020:", [(r['university'], r['major_name'], r['action']) for r in bj if r['is_cs_ai_info']][:8])


if __name__ == "__main__":
    main()
