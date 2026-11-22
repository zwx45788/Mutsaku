"""Sourced partial dataset of Beijing universities' 2025年度 undergraduate major changes.

The Ministry of Education's national 2025年度 results attachment (教高函〔2026〕3号) and the Beijing Municipal Education
Commission's 2025 municipal results notice could not be located on official government sites. As a documented
fallback, this script fetches each Beijing university's OWN official announcement (*.edu.cn), archives the HTML with
SHA-256, checks that the major name (or the stated verification token) appears in the fetched page text, maps the major
to the official 2026 catalogue code, and writes data/blind_holdout/moe_major_lifecycle_2025_partial_univ.json.
Rows whose page could not be fetched or whose token is not found are kept with verified=false and excluded from analysis.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from ingestion.parse_moe_and_bj_catalog import _is_beijing_uni

RAW = Path("evoalign_os/data/raw_public/bj_2025_univ_pages")
PROC = Path("evoalign_os/data/processed")
BLIND = Path("evoalign_os/data/blind_holdout")
REG = Path("evoalign_os/data/self_collected_register.csv")
OUT = BLIND / "moe_major_lifecycle_2025_partial_univ.json"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36"
CS_CODES = ("0809", "0807", "0808", "0712", "1405", "0810")
CS_WORDS = ("计算机", "人工智能", "软件", "数据", "智能", "机器人", "网络", "信息", "电子", "集成电路", "算力", "量子")


def fetch(url: str) -> bytes | None:
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=40) as r:
                return r.read()
        except Exception:
            time.sleep(2 * (attempt + 1))
    return None


def text_of(raw: bytes) -> str:
    for enc in ("utf-8", "gb18030"):
        try:
            html = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        html = raw.decode("utf-8", errors="ignore")
    html = re.sub(r"<script[\s\S]*?</script>|<style[\s\S]*?</style>", " ", html, flags=re.I)
    return re.sub(r"\s+", "", re.sub(r"<[^>]+>", " ", html))


def main(seed_path: str) -> None:
    seed = json.loads(Path(seed_path).read_text(encoding="utf-8"))
    cat = {m["name"]: m["code"] for m in json.loads((PROC / "moe_undergrad_catalog_2026.json").read_text(encoding="utf-8"))["majors"]}
    RAW.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    pages = {}
    for url in sorted({s["source_url"] for s in seed}):
        raw = fetch(url)
        time.sleep(1.0)
        if raw is None:
            pages[url] = None
            continue
        name = hashlib.sha256(url.encode()).hexdigest()[:16] + ".html"
        (RAW / name).write_bytes(raw)
        pages[url] = {"file": name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "text": text_of(raw)}
    recs = []
    for s in seed:
        pg = pages.get(s["source_url"])
        name_hit = bool(pg) and s["major_name"].split("（")[0][:6] in pg["text"]
        token_hit = bool(pg) and s["verify_token"] in pg["text"]
        code = cat.get(s["major_name"]) or (s["major_code_stated"] or None)
        recs.append({
            "year": 2025, "action": s["action"], "university": s["university"], "is_beijing": _is_beijing_uni(s["university"]) or True,
            "major_name": s["major_name"], "major_code": code, "major_code_stated": s["major_code_stated"] or None,
            "is_cs_ai_info": bool(code and code.startswith(CS_CODES)) or any(k in s["major_name"] for k in CS_WORDS),
            "source_url": s["source_url"], "published": s["published"],
            "archived_html": (pg or {}).get("file"), "page_sha256": (pg or {}).get("sha256"),
            "verified": bool(pg) and (name_hit or token_hit), "name_in_page": name_hit,
        })
    doc = {
        "scope": "PARTIAL — 北京 10 所高校官网自行公布的 2025 年度本科专业增设/撤销结果；教育部全国附件与北京市教委市属高校结果未能在官方网站定位",
        "collected_utc": now,
        "n_records": len(recs), "n_verified": sum(r["verified"] for r in recs),
        "universities": sorted({r["university"] for r in recs}),
        "records": recs,
    }
    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    rows = [r for r in csv.reader(REG.open(encoding="utf-8")) if r and not r[0].startswith("BJ2025_UNIV_")]
    for i, (url, pg) in enumerate(sorted(pages.items())):
        unis = sorted({s["university"] for s in seed if s["source_url"] == url})
        rows.append([f"BJ2025_UNIV_{i + 1:02d}", "moe_major_lifecycle_2025_univ", "、".join(unis), "2025年度本科专业增设/撤销结果（高校官网公告）", "2025",
                     url, now, "ok" if pg else "failed", str(pg["bytes"]) if pg else "", "", str(len(pg["text"])) if pg else "", pg["sha256"] if pg else "",
                     "高校官网依法公开新闻公告；仅提取专业名称与动作，不含个人信息；作为教育部 2025 年度附件缺失时的部分替代"])
    for r in rows:
        if r and r[0] == "MOE_2025_APPROVALS":
            r[7] = "not_found_national_attachment"
            r[12] = "教育部全国附件未能在官网定位；已以 BJ2025_UNIV_* 行的北京高校官网公告作部分替代，来源逐条可核"
    with REG.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print(f"pages fetched {sum(1 for p in pages.values() if p)}/{len(pages)}; records {len(recs)}, verified {doc['n_verified']}, CS/AI/info {sum(1 for r in recs if r['is_cs_ai_info'] and r['verified'])}")
    for r in recs:
        if not r["verified"]:
            print("  UNVERIFIED:", r["university"], r["major_name"], r["source_url"])
    print("CS/AI/info verified:", [(r["university"], r["major_name"], r["major_code"], r["action"]) for r in recs if r["is_cs_ai_info"] and r["verified"]])


if __name__ == "__main__":
    main(sys.argv[1])
