"""最小输入→输出演示：岗位文本 → 48 项标准技能 + 岗位族，并与 sample_jd_output.csv 逐行比对。"""
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
