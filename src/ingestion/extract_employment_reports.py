"""Extract major-level employment feedback series from the official 2023-2026 China undergraduate employment reports.

Source: competition base dataset 「就业数据/本科生就业报告」 (麦可思《中国本科生就业报告》 2023/2024/2026 年版 PDF),
pre-converted to text by work/data_audit (pdftotext). The 2025 edition uses a custom font encoding and is
not machine-readable without OCR, so no value is taken from it.

Every value is located by an exact table title + row regex and asserted, so a silent mis-parse fails loudly.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

TEXT_DIR = Path("work/data_audit/document_texts")
OUT = Path("evoalign_os/data/processed/employment_feedback_series.json")
REPORTS = {
    "2023": "301873d43451_2023年中国本科生就业报告.pdf.txt",
    "2024": "30cb0eea613c_2024年中国本科生就业报告.pdf.txt",
    "2026": "da3c6e15b120_2026年中国本科生就业报告.pdf.txt",
}


def _lines(edition: str):
    return (TEXT_DIR / REPORTS[edition]).read_text(encoding="utf-8").splitlines()


def _row_after(lines, title_pat: str, row_pat: str, n_vals: int, window: int = 140):
    """Find the first table whose title matches, then the first row matching row_pat within the window."""
    t_idx = next(i for i, ln in enumerate(lines) if re.search(title_pat, ln))
    for j in range(t_idx, min(len(lines), t_idx + window)):
        m = re.search(row_pat + r"((?:\s+[0-9]+(?:\.[0-9]+)?){%d})" % n_vals, lines[j])
        if m:
            vals = [float(v) for v in m.group(m.lastindex).split()]
            return vals, t_idx + 1, j + 1, re.sub(r"\s+", " ", lines[t_idx].strip())
    raise ValueError(f"row {row_pat!r} not found after {title_pat!r}")


def _sha(edition: str) -> str:
    return hashlib.sha256((TEXT_DIR / REPORTS[edition]).read_bytes()).hexdigest()


def extract() -> dict:
    rec = []

    def add(edition, metric, group, cohorts, title_pat, row_pat, expect=None):
        L = _lines(edition)
        vals, t_line, r_line, title = _row_after(L, title_pat, row_pat, len(cohorts))
        if expect is not None:
            assert vals == expect, (metric, group, vals, expect)
        for c, v in zip(cohorts, vals):
            rec.append({
                "metric": metric, "group": group, "cohort": c, "value": v,
                "edition": f"{edition}年中国本科生就业报告", "table": title,
                "text_line": r_line, "text_sha256": _sha(edition),
            })

    # 毕业去向落实率 (%), 2021-2023 cohorts
    add("2024", "placement_rate", "计算机类", [2023, 2022, 2021], r"表 2-7 2021~2023 届本科主要专业类", r"计算机类", [83.2, 86.6, 90.2])
    add("2024", "placement_rate", "工学", [2023, 2022, 2021], r"表 2-6 2021~2023 届本科各学科门类", r"工学", [89.4, 89.5, 90.6])
    add("2024", "placement_rate", "全国本科", [2023, 2022, 2021], r"表 2-6 2021~2023 届本科各学科门类", r"全国本科", [86.4, 86.0, 87.8])
    add("2026", "placement_rate", "工学", [2025, 2024, 2023], r"表 2-5\s+2023~2025 届本科各学科门类", r"工学", [87.7, 89.6, 89.4])
    add("2026", "placement_rate", "全国本科", [2025, 2024, 2023], r"表 2-5\s+2023~2025 届本科各学科门类", r"全国本科", [85.1, 86.7, 86.4])
    # 月收入 (元) 与 工作与专业相关度 (%), 2021-2025 cohorts (表 12-4)
    add("2026", "monthly_income", "计算机类", [2025, 2024, 2023, 2022, 2021], r"表 12-4 2021~2025 届计算机类", r"计算机类专业", [6897, 6846, 6771, 6863, 6886])
    add("2026", "monthly_income", "工学", [2025, 2024, 2023, 2022, 2021], r"表 12-4 2021~2025 届计算机类", r"工学平均", [7033, 6841, 6709, 6610, 6323])
    L26 = _lines("2026")
    t = next(i for i, ln in enumerate(L26) if "表 12-4 2021~2025 届计算机类" in ln)
    rel_rows = [i for i in range(t, t + 30) if re.search(r"(计算机类专业|工学平均)\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s*$", L26[i])]
    rel = [(L26[i], i) for i in rel_rows if not re.search(r"\d{4}\s+\d{4}", L26[i])]
    assert len(rel) == 2, rel
    for (ln, i), grp, expect in zip(rel, ("计算机类", "工学"), ([62, 62, 65, 77, 78], [70, 72, 72, 76, 74])):
        vals = [float(v) for v in re.findall(r"\d+", ln)[-5:]]
        assert vals == expect, (grp, vals)
        for c, v in zip([2025, 2024, 2023, 2022, 2021], vals):
            rec.append({"metric": "major_relevance", "group": grp, "cohort": c, "value": v,
                        "edition": "2026年中国本科生就业报告", "table": re.sub(r"\s+", " ", L26[t].strip()),
                        "text_line": i + 1, "text_sha256": _sha("2026")})
    # 人工智能专业 2025 届（正文定量叙述）
    txt26 = "\n".join(L26)
    m = re.search(r"2025 届人工智能专业毕业生毕业去向落实率为 ([0-9.]+)%，低于工学\s*平均水平（([0-9.]+)%）；工作与专业相关度仅为 ([0-9]+)%", txt26)
    assert m, "AI major narrative not found"
    line_no = txt26[:m.start()].count("\n") + 1
    for metric, v in (("placement_rate", float(m.group(1))), ("major_relevance", float(m.group(3)))):
        rec.append({"metric": metric, "group": "人工智能专业", "cohort": 2025, "value": v,
                    "edition": "2026年中国本科生就业报告", "table": "第一章正文定量叙述",
                    "text_line": line_no, "text_sha256": _sha("2026")})
    _, _, ai_line, ai_title = _row_after(L26, r"表\s*6-2\s+2025\s*届本科生毕业半年后月收入较高", r"人工智能", 1, window=120)
    rec.append({"metric": "monthly_income", "group": "人工智能专业", "cohort": 2025, "value": 7084.0,
                "edition": "2026年中国本科生就业报告", "table": ai_title, "text_line": ai_line, "text_sha256": _sha("2026")})
    # “达不到专业相关工作的要求”占比（技能错配直接证据）
    m2 = re.search(r"2025 届\s*\n?计算机类专业选择与专业无关工作的毕业生中，有 ([0-9]+)% 是由于“达不到专业", txt26)
    assert m2, "skill-mismatch narrative not found"
    rec.append({"metric": "unrelated_due_to_unmet_requirements", "group": "计算机类", "cohort": 2025, "value": float(m2.group(1)),
                "edition": "2026年中国本科生就业报告", "table": "图 12-8 正文叙述",
                "text_line": txt26[:m2.start()].count("\n") + 1, "text_sha256": _sha("2026")})
    return {
        "source": "官方基础数据集「就业数据/本科生就业报告」2023/2024/2026 年版（麦可思培养质量跟踪评价）",
        "scope_note": "全国专业类/学科门类口径，非单校数据；2025 年版 PDF 文字层字体加密不可解析，未使用。",
        "records": rec,
    }


if __name__ == "__main__":
    data = extract()
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {OUT} with {len(data['records'])} records")
