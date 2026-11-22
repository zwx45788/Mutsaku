"""Curated crosswalk from the 48 canonical skills / 4 career tracks to ESCO v1 concepts, verified against the official API.

Each skill lists candidate ESCO preferred labels chosen by the team; every label is looked up through the ESCO search
API and accepted only if a concept with exactly that preferred label (case-insensitive) is returned. The resulting real
concept URIs replace the earlier placeholder codes in dynamic_kg.json["skills"][*]["esco"].
Outputs data/processed/esco_alignment.json and a provenance row in data/self_collected_register.csv.
"""
from __future__ import annotations

import csv
import hashlib
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

API = "https://ec.europa.eu/esco/api/search"
PROC = Path("evoalign_os/data/processed")
REG = Path("evoalign_os/data/self_collected_register.csv")

CROSSWALK = {
    "discrete_math": ["set theory", "mathematics", "algebra"],
    "linear_algebra_opt": ["algebra", "geometry", "mathematics"],
    "prob_stats": ["statistics", "probability theory", "apply statistical analysis techniques"],
    "dsa": ["algorithms", "computer science"],
    "comp_arch": ["hardware architectures", "computer engineering", "maintain computer hardware"],
    "os_kernel": ["operating systems", "use concurrent programming", "computer engineering"],
    "linux_shell": ["use scripting programming", "operating systems", "Kali Linux"],
    "comp_net": ["ICT communications protocols", "ICT networking hardware"],
    "compiler_pl": ["computer programming", "ICT debugging tools", "use functional programming"],
    "parallel_gpu": ["use concurrent programming", "parallel computing"],
    "c_cpp": ["C++", "computer programming"],
    "python_eng": ["Python (computer programming)"],
    "java_spring": ["Java (computer programming)", "use object-oriented programming"],
    "golang_rust": ["use concurrent programming", "use functional programming"],
    "web_frontend": ["JavaScript", "web programming"],
    "mobile_harmony": ["mobile operating systems", "mobile device software frameworks"],
    "soft_eng_arch": ["software architecture models", "use software design patterns"],
    "qa_testing": ["tools for ICT test automation", "execute software tests"],
    "ai_native_se": ["use automatic programming", "principles of artificial intelligence"],
    "dist_systems": ["distributed computing"],
    "cloud_k8s": ["cloud technologies"],
    "devops_sre": ["DevOps", "tools for software configuration management"],
    "sql_rdbms": ["SQL", "database management systems"],
    "nosql_vector_db": ["NoSQL", "maintain database"],
    "big_data_eng": ["analyse big data", "data models"],
    "data_mining_bi": ["data mining", "business intelligence"],
    "ml_classic": ["utilise machine learning", "apply statistical analysis techniques"],
    "deep_learning": ["utilise machine learning", "principles of artificial intelligence"],
    "nlp_text": ["natural language processing", "information extraction"],
    "cv_multimodal": ["computer vision", "optical character recognition software"],
    "llm_posttrain": ["utilise machine learning", "natural language processing"],
    "agent_rag": ["principles of artificial intelligence", "information extraction"],
    "rl_decision": ["utilise machine learning", "decision support systems"],
    "mlops_eval": ["utilise machine learning", "ICT performance analysis methods"],
    "embedded_mcu": ["embedded systems"],
    "iot_edge": ["Internet of Things", "smart devices"],
    "robotics_embodied": ["robotics", "robotic components"],
    "control_signal": ["control engineering", "signal processing"],
    "cyber_sec": ["cyber security", "ICT encryption"],
    "ai_safety_privacy": ["data protection", "ICT security legislation"],
    "graphics_game": ["digital game creation systems", "operate 3D computer graphics software"],
    "blockchain_fintech": ["blockchain consensus mechanisms", "blockchain openness", "financial engineering"],
    "hci_product": ["human-computer interaction", "plan product management"],
    "tech_comm_ethics": ["apply technical communication skills", "follow ethical code of conduct"],
    "legacy_erp_vb": ["Visual Studio .NET", "manage standard enterprise resource planning system"],
    "open_source_collab": ["Open source model", "tools for software configuration management"],
    "eda_ic_design": ["design integrated circuits", "integrated circuits"],
    "bio_sci_computing": ["computer science", "bioinformatics"],
}
OCCUPATIONS = {
    "fullstack_backend": ["software developer", "web developer", "software architect"],
    "cloud_ops_sec": ["ICT system administrator", "ICT network engineer", "software tester", "ICT security engineer"],
    "ai_data": ["data scientist", "ICT intelligent systems designer", "data analyst"],
    "systems_edge": ["embedded system designer", "embedded systems software developer", "robotics engineer"],
}


def search(text: str, typ: str, limit: int = 10):
    q = urllib.parse.urlencode({"text": text, "type": typ, "language": "en", "limit": limit, "full": "false"})
    last = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(f"{API}?{q}", timeout=30) as r:
                data = json.loads(r.read().decode("utf-8"))
            return data.get("_embedded", {}).get("results", []), None
        except Exception as e:  # noqa: BLE001
            last = repr(e)
            time.sleep(2.0 * (attempt + 1))
    return [], last


def slim(hit):
    return {
        "uri": hit.get("uri"), "label_en": hit.get("title"), "label_zh": (hit.get("preferredLabel") or {}).get("zh"),
        "skill_type": [x.rsplit("/", 1)[-1] for x in hit.get("hasSkillType", [])],
        "reuse_level": [x.rsplit("/", 1)[-1] for x in hit.get("hasReuseLevel", [])],
        "isced_f": [x.rsplit("/", 1)[-1] for x in hit.get("broaderHierarchyConcept", []) if "isced-f" in x],
        "isco": [x.rsplit("/", 1)[-1] for x in hit.get("broaderIscoGroup", [])] or None,
    }


def verify(label: str, typ: str):
    hits, err = search(label, typ)
    time.sleep(0.3)
    exact = next((h for h in hits if (h.get("title") or "").strip().lower() == label.strip().lower()), None)
    return (slim(exact) if exact else None), err


def main():
    kg_path = PROC / "dynamic_kg.json"
    kg = json.loads(kg_path.read_text(encoding="utf-8"))
    out = {"source": "ESCO v1 (European Skills, Competences, Qualifications and Occupations) — official REST API " + API,
           "retrieved_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "method": "team-curated crosswalk of ESCO preferred labels; each label verified to exist by exact (case-insensitive) title match in the API search results",
           "skills": {}, "tracks": {}, "unverified_labels": []}
    calls = 0
    for s in kg["skills"]:
        sid = s["id"]
        rec = {"en": s["en"], "zh": s["zh"], "cs2023_ka": s.get("cs2023_ka"), "concepts": []}
        for label in CROSSWALK[sid]:
            hit, err = verify(label, "skill")
            calls += 1
            if hit:
                if hit["uri"] not in [c["uri"] for c in rec["concepts"]]:
                    rec["concepts"].append(hit)
            else:
                out["unverified_labels"].append({"skill": sid, "label": label, "error": err})
        out["skills"][sid] = rec
        s["esco"] = [c["uri"] for c in rec["concepts"]]
        s["esco_labels"] = [c["label_en"] for c in rec["concepts"]]
        s.pop("esco_match_quality", None)
        print(f"{sid:20s} -> {' | '.join(c['label_en'] for c in rec['concepts']) or 'NO VERIFIED CONCEPT'}")
    for tk, labels in OCCUPATIONS.items():
        occ = []
        for label in labels:
            hit, err = verify(label, "occupation")
            calls += 1
            if hit:
                occ.append(hit)
            else:
                out["unverified_labels"].append({"track": tk, "label": label, "error": err})
        out["tracks"][tk] = occ
        print(f"[track] {tk:18s} -> {' | '.join(o['label_en'] + '(' + ','.join(o['isco'] or []) + ')' for o in occ)}")
    covered = sum(1 for v in out["skills"].values() if v["concepts"])
    out["coverage"] = {"skills_with_verified_concept": covered, "skills_total": len(kg["skills"]),
                       "verified_skill_concepts": sum(len(v["concepts"]) for v in out["skills"].values()),
                       "verified_occupations": sum(len(v) for v in out["tracks"].values()),
                       "unverified_labels": len(out["unverified_labels"]), "api_calls": calls}
    kg["esco_alignment"] = {"file": "esco_alignment.json", **out["coverage"], "retrieved_utc": out["retrieved_utc"]}
    ap = PROC / "esco_alignment.json"
    ap.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    kg_path.write_text(json.dumps(kg, ensure_ascii=False, indent=1), encoding="utf-8")
    rows = [r for r in csv.reader(REG.open(encoding="utf-8")) if r and r[0] != "STD_ESCO_V1"]
    raw = ap.read_bytes()
    rows.append(["STD_ESCO_V1", "standard_ontology", "European Commission (ESCO)", "ESCO v1 技能与职业本体对照(48技能+4岗位族)", "2026",
                 API, out["retrieved_utc"], "ok", str(len(raw)), "", str(len(raw.decode("utf-8"))), hashlib.sha256(raw).hexdigest(),
                 "欧盟官方公开 API；仅检索技能/职业概念元数据，不含个人信息；调用间隔 ≥ 0.3 秒"])
    with REG.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print(json.dumps(out["coverage"], ensure_ascii=False))
    print("unverified:", [(u.get("skill") or u.get("track"), u["label"], u["error"]) for u in out["unverified_labels"]])


if __name__ == "__main__":
    main()
