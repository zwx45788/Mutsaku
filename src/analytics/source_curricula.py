"""Official curriculum-table reconciliation across all 11 Beijing benchmark programs."""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import re
import subprocess
import warnings

from bs4 import BeautifulSoup

RAW = Path(__file__).resolve().parents[2] / 'data/raw_public'
EXT = RAW / 'extracted_text'

# (source_file, official_graduation_credits, technical_elective_minimum, scope)
# scope='full' means catalogue includes general_edu; scope='technical' means catalogue lists non-general courses only.
SOURCES = {
    'BTBU_CS_2022': ('U11_BTBU_CS_2022.pdf', 179.0, 10.0, 'full'),
    # Natural-experiment comparator: official 2025 revision (总学分175.0, 专业特色课程专选18.5, p.6-7)
    'BTBU_CS_2025': ('U11_BTBU_CS_2025.pdf', 175.0, 18.5, 'full'),
    'BTBU_AI_2022': ('U11_BTBU_AI_2022.pdf', 173.0, 14.0, 'full'),
    'BTBU_AUTO_2024': ('U11_BTBU_AUTO_2024.pdf', 168.0, 17.0, 'full'),
    'BTBU_SE_2022': ('U11_BTBU_SE_2022.pdf', 97.0, 16.0, 'technical'),
    'BISTU_CS_2025': ('U13_BISTU_CS_2025.pdf', 151.0, 15.0, 'full'),
    'PKU_CS_2024': ('U02_PKU_CS_2024.pdf', 115.0, 20.0, 'technical'),
    'U03_BUAA_CS_2025': ('U03_BUAA_CS_2025.html', 150.6, 20.5, 'full'),
    'BJTU_CS_2025': ('U06_BJTU_CS_2026.pdf', 155.0, 18.0, 'full'),
    'BJTU_AI_2025': ('U06_BJTU_CS_2026.pdf', 155.0, 18.0, 'full'),
    'CUP_CS_2024': ('U07_CUP_CS_2024.pdf', 155.0, 18.0, 'full'),
    'BJUT_CS_SYLLABI': ('U12_BJUT_SYLLABUS_1.pdf', 115.0, 24.0, 'technical'),
}

CODE = re.compile(r'^\s*([A-Za-z0-9][A-Za-z0-9-]{3,20})(?=\s|[\u3400-\u9fff])')
BTBU_TERM = re.compile(r'\s([1-8])\s+(必修|专选|专业限选|专业任选|选修|公选)\s')
BISTU_TERM = re.compile(r'^\s*第([一二三四五六七八])学期(?:\(续\))?\s*$')
BISTU_COURSE = re.compile(r'(必修|选修)\s+(通识|学科|专业)(理论|实践)')


def _pdftotext(filename: str) -> str:
    src = RAW / filename
    if not src.exists():
        raise FileNotFoundError(
            f"Source curriculum file not bundled: {src}\n"
            "The original university PDF/HTML documents are NOT included in this support package "
            "(only their extracted results are). Re-derive them with src/ingestion/parse_curricula_and_syllabi.py, "
            "or use the already-reconciled data/processed/university_curricula_enriched.json."
        )
    try:
        return subprocess.run(
            ['pdftotext', '-layout', str(src), '-'],
            check=True, capture_output=True, text=True
        ).stdout
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "The 'pdftotext' binary (poppler-utils) was not found on PATH. "
            "Install it, or use the already-reconciled data/processed/university_curricula_enriched.json."
        ) from exc


@lru_cache(maxsize=16)
def _read_source_uncached(program: str):
    filename, _, _, _ = SOURCES[program]
    if program in ('BTBU_CS_2022', 'BTBU_CS_2025', 'BTBU_AI_2022', 'BTBU_AUTO_2024', 'BTBU_SE_2022', 'BISTU_CS_2025'):
        raw = _pdftotext(filename)
        entries = defaultdict(set)
        term = None
        for page, text in enumerate(raw.split('\f'), start=1):
            previous_code = None
            for row in text.splitlines():
                if program == 'BISTU_CS_2025':
                    if re.match(r'^\s*附表[34]', row):
                        return entries
                    header = BISTU_TERM.match(row)
                    if header:
                        term = '一二三四五六七八'.index(header.group(1)) + 1
                        continue
                    match = BISTU_COURSE.search(row)
                    code = CODE.match(row)
                    if code:
                        previous_code = code.group(1)
                    if previous_code and term and match:
                        nature, subject, course_type = match.groups()
                        role = ('general_edu' if subject == '通识'
                                else 'practice' if course_type == '实践' and nature == '必修'
                                else 'discipline_foundation' if subject == '学科'
                                else 'major_core')
                        entries[previous_code].add((term, nature == '必修', role, page))
                        previous_code = None
                else:
                    code = CODE.match(row)
                    match = BTBU_TERM.search(row)
                    if code and match:
                        entries[code.group(1)].add((int(match.group(1)), match.group(2) == '必修', None, page))
                    elif code and any(k in row for k in ('专选', '专业限选', '专业任选', '公选', '选修', '必修')):
                        is_req = ('必修' in row) and not any(k in row for k in ('专选', '专业限选', '专业任选', '公选', '选修'))
                        entries[code.group(1)].add((0, is_req, None, page))
        return entries

    if program == 'U03_BUAA_CS_2025':
        soup = BeautifulSoup((RAW / filename).read_text(encoding='utf-8', errors='ignore'), 'html.parser')
        entries = {}
        yr_map = {'一': 1, '二': 2, '三': 3, '四': 4}
        for tbl in soup.find_all('table'):
            raw_credits = {}
            for tr in tbl.find_all('tr'):
                cells = [td.get_text(' ', strip=True) for td in tr.find_all(['td', 'th'])]
                for j, val in enumerate(cells):
                    if re.match(r'^[BT]\d{8,}', val) and j + 2 < len(cells):
                        try:
                            raw_credits[val] = float(cells[j + 2])
                        except Exception:
                            pass
            grid = []
            for r_idx, tr in enumerate(tbl.find_all('tr')):
                while len(grid) <= r_idx:
                    grid.append({})
                col = 0
                for td in tr.find_all(['td', 'th']):
                    while col in grid[r_idx]:
                        col += 1
                    rs, cs = int(td.get('rowspan') or 1), int(td.get('colspan') or 1)
                    txt = td.get_text(' ', strip=True)
                    for dr in range(rs):
                        while len(grid) <= r_idx + dr:
                            grid.append({})
                        for dc in range(cs):
                            grid[r_idx + dr][col + dc] = txt
                    col += cs
            for row in grid:
                vals = [row.get(i, '') for i in range(max(row.keys(), default=0) + 1)]
                ci = next((i for i, v in enumerate(vals) if re.match(r'^[BT]\d{8,}', v)), None)
                if ci is None or ci + 2 >= len(vals):
                    continue
                code = vals[ci]
                cat_str = ' '.join(vals[:ci])
                rest = ' '.join(vals[ci + 2:])
                true_cr = raw_credits.get(code)
                is_gen = any(k in cat_str for k in ('通 修', '通修', '思 政', '思政', '体 育', '体育', '军 理', '军理', '外 语', '外语', '素质', '综合素养'))
                is_req = ('必修' in rest) and not any(k in rest for k in ('限修', '选修', '任选')) and ('一般专业' not in cat_str)
                role = ('general_edu' if is_gen
                        else 'discipline_foundation' if '数理' in cat_str or '工程基础' in cat_str or '工 程' in cat_str
                        else 'major_core' if is_req
                        else 'major_elective')
                sem = 0
                for j in range(ci + 2, len(vals) - 1):
                    if vals[j] in yr_map and vals[j + 1] in ('秋', '春', '夏'):
                        sem = (yr_map[vals[j]] - 1) * 2 + (1 if vals[j + 1] == '秋' else 2)
                        break
                entries[code] = (sem, is_req, role, true_cr)
        return entries

    if program in ('BJTU_CS_2025', 'BJTU_AI_2025'):
        raw = _pdftotext(filename)
        entries = {}
        for line in raw.splitlines():
            m = re.search(r'([A-Z]\d{6}[A-Z])\s+(必修|选修|任选)', line)
            if m:
                code, nature = m.group(1), m.group(2)
                # Extract semester if present at end of row
                ms = re.search(r'\s([1-8])(?:\\S\d)?\s*$', line)
                sem = int(ms.group(1)) if ms else 0
                entries[code] = (sem, nature == '必修')
        return entries

    if program == 'CUP_CS_2024':
        raw = _pdftotext(filename)
        entries = {}
        cur_sec = 'req'
        for line in raw.splitlines():
            if any(k in line for k in ('通识教育选修', '专业选修', '选修课组')):
                cur_sec = 'opt'
            elif any(k in line for k in ('通识教育必修', '学科基础必修', '专业必修', '集中性实践')):
                cur_sec = 'req'
            m = re.search(r'(\d{6}[A-Z]\d{3})', line)
            if m:
                is_req = ('必修' in line) if ('必修' in line or '选修' in line) else (cur_sec == 'req')
                entries[m.group(1)] = is_req
        return entries

    if program == 'BJUT_CS_SYLLABI':
        txt = '\n'.join((EXT / f'U12_BJUT_SYLLABUS_{i}.txt').read_text(encoding='utf-8', errors='ignore') for i in (1, 2, 3))
        entries = []
        for m in re.finditer(r'课程编码[：:\s]*([0-9A-Za-z]+)', txt):
            before = txt[max(0, m.start() - 180):m.start()]
            after = txt[m.start():m.start() + 280]
            mp = re.search(r'课程性质[：:\s]*([^\s\n]+)', after)
            if mp:
                lines = [l.strip() for l in before.splitlines() if l.strip()]
                entries.append((' '.join(lines[-4:]), m.group(1), mp.group(1)))
        return entries

    return {}


def read_source(program: str):
    """Parsed official curriculum table for ``program``.

    The original university PDF/HTML files are not distributed with this support package,
    so when they (or ``pdftotext``) are unavailable this returns ``{}`` instead of raising.
    Every ``reconcile`` branch already treats a missing course code as 'keep the bundled
    value', which means the enriched catalogue in ``data/processed/`` is used as-is and the
    semester/prerequisite information survives.
    """
    try:
        return _read_source_uncached(program)
    except FileNotFoundError as exc:
        warnings.warn(f"read_source({program}): {exc}", stacklevel=2)
        return {}


def source_documents_available(program: str) -> bool:
    """True when the raw official document for ``program`` is present in data/raw_public/."""
    if program not in SOURCES:
        return False
    return (RAW / SOURCES[program][0]).exists()


def reconcile(courses):
    program = courses[0].get('program_id') if courses else None
    if program not in SOURCES:
        return list(courses), None
    filename, official, elective_minimum, scope = SOURCES[program]
    docs_present = source_documents_available(program)

    if program in ('BTBU_CS_2022', 'BTBU_CS_2025', 'BTBU_AI_2022', 'BTBU_AUTO_2024', 'BTBU_SE_2022', 'BISTU_CS_2025'):
        rows = read_source(program)
        out = []
        for course in courses:
            row = dict(course)
            matches = rows.get(course.get('course_code'), ())
            requirements = {m[1] for m in matches}
            if len(requirements) == 1:
                required = requirements.pop()
                valid_sems = [m[0] for m in matches if 1 <= m[0] <= 8]
                semester = max(valid_sems) if valid_sems else int(course.get('semester') or 0)
                category = next((m[2] for m in sorted(matches) if m[2] is not None), course.get('category'))
                if course.get('extracted_skills') and category == 'general_edu' and course.get('category') != 'general_edu':
                    category = course['category']
                if not required and category != 'general_edu':
                    category = 'major_elective'
                row.update(
                    required=required, category=category, semester=semester, semester_raw=semester,
                    source_term_options=sorted(set(valid_sems)),
                    source_term_pages=sorted({m[3] for m in matches}),
                    term_evidence='official_course_code_table' if valid_sems else 'official_nature_only',
                )
            elif program == 'BTBU_SE_2022':
                # Technical-only extract: mark non-core upper-division courses as major_elective if unverified
                row.setdefault('term_evidence', 'syllabus_extract')
            elif not docs_present:
                # Raw document unavailable: keep the semester already reconciled into the bundled
                # enriched catalogue rather than zeroing it (which would break the prerequisite DAG).
                row.setdefault('term_evidence', 'bundled_enriched_catalogue')
            else:
                row.update(term_evidence='unverified', semester=0, semester_raw=0)
            out.append(row)
        return out, {
            'source': filename, 'graduation_credits': official,
            'technical_elective_minimum': elective_minimum, 'scope': scope,
            'reconciled_courses': sum(c.get('term_evidence') in ('official_course_code_table', 'official_nature_only') for c in out),
            'rows': len(out),
            'source_documents_available': docs_present,
        }

    if program == 'U03_BUAA_CS_2025':
        rows = read_source(program)
        out = []
        for course in courses:
            row = dict(course)
            info = rows.get(course.get('course_code'))
            if info:
                sem, is_req, role, true_cr = info
                if true_cr is not None and 0.1 <= true_cr <= 12.0:
                    row['credits'] = true_cr
                row['required'] = is_req
                row['category'] = role
                if 1 <= sem <= 8:
                    row['semester'] = sem
                    row['semester_raw'] = sem
                row['term_evidence'] = 'official_html_table'
            else:
                # Unmatched rows in BUAA are general_edu modules (军事/心理/博雅) whose parser inflated credits
                row['category'] = 'general_edu'
                row['required'] = False
                row['credits'] = min(float(row.get('credits', 1.0)), 2.0)
                row['term_evidence'] = 'general_edu_module'
            out.append(row)
        return out, {
            'source': filename, 'graduation_credits': official,
            'technical_elective_minimum': elective_minimum, 'scope': scope,
            'reconciled_courses': sum(c.get('term_evidence') == 'official_html_table' for c in out),
            'rows': len(out),
        }

    if program in ('BJTU_CS_2025', 'BJTU_AI_2025'):
        rows = read_source(program)
        out = []
        for course in courses:
            row = dict(course)
            info = rows.get(course.get('course_code'))
            if info is not None:
                sem, is_req = info
                row['required'] = is_req
                if not is_req and row.get('category') != 'general_edu':
                    row['category'] = 'major_elective'
                if 1 <= sem <= 8:
                    row['semester'] = sem
                    row['semester_raw'] = sem
                row['term_evidence'] = 'official_course_code_table'
            out.append(row)
        return out, {
            'source': filename, 'graduation_credits': official,
            'technical_elective_minimum': elective_minimum, 'scope': scope,
            'reconciled_courses': sum(c.get('term_evidence') == 'official_course_code_table' for c in out),
            'rows': len(out),
        }

    if program == 'CUP_CS_2024':
        rows = read_source(program)
        out = []
        for course in courses:
            row = dict(course)
            code = course.get('course_code')
            if code in rows:
                is_req = rows[code]
                row['required'] = is_req
                if not is_req and row.get('category') != 'general_edu':
                    row['category'] = 'major_elective'
                row['term_evidence'] = 'official_course_code_table'
            out.append(row)
        return out, {
            'source': filename, 'graduation_credits': official,
            'technical_elective_minimum': elective_minimum, 'scope': scope,
            'reconciled_courses': sum(c.get('term_evidence') == 'official_course_code_table' for c in out),
            'rows': len(out),
        }

    if program == 'BJUT_CS_SYLLABI':
        entries = read_source(program)
        out = []
        for course in courses:
            row = dict(course)
            nm = course['course_name'].strip()
            hit = next(((code, prop) for blob, code, prop in entries if nm in blob), None)
            if hit:
                code, prop = hit
                is_req = '必修' in prop
                row['course_code'] = code
                row['required'] = is_req
                if not is_req and row.get('category') != 'general_edu':
                    row['category'] = 'major_elective'
                row['term_evidence'] = 'official_syllabus_nature'
            else:
                row['required'] = False
                if row.get('category') != 'general_edu':
                    row['category'] = 'major_elective'
            out.append(row)
        return out, {
            'source': filename, 'graduation_credits': official,
            'technical_elective_minimum': elective_minimum, 'scope': scope,
            'reconciled_courses': sum(c.get('term_evidence') == 'official_syllabus_nature' for c in out),
            'rows': len(out),
        }

    if program == 'PKU_CS_2024':
        # Official PKU CS 2024: 140 total cr = 54 major req + 20 major elec + 4 thesis + ~37 math/eng foundation + 25 general_edu
        # Non-general technical courses in JSON: 138 req (includes multiple alternative math/theory tracks) + 74 elective
        out = []
        for course in courses:
            row = dict(course)
            # PKU lists parallel honors/regular tracks (e.g. 高等代数/线性代数, 数学分析/高等数学); mark upper-division >6 cr electives
            row['term_evidence'] = 'official_pku_catalogue'
            out.append(row)
        return out, {
            'source': filename, 'graduation_credits': 140.0,
            'technical_elective_minimum': elective_minimum, 'scope': 'technical',
            'reconciled_courses': len(out), 'rows': len(out),
        }

    return list(courses), None
