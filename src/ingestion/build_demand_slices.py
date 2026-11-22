"""Build strict temporal demand slices from work/modern_data/catalog.duckdb.
Physically isolates 2025 holdout and 2026Q1 drift into evoalign_os/data/blind_holdout/.
Preserves full de-identified technical job skill texts, salary, education, experience, and employer keys.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import duckdb

DB_PATH = "work/modern_data/catalog.duckdb"
PROC_DIR = Path("evoalign_os/data/processed")
BLIND_DIR = Path("evoalign_os/data/blind_holdout")
PROC_DIR.mkdir(parents=True, exist_ok=True)
BLIND_DIR.mkdir(parents=True, exist_ok=True)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    con = duckdb.connect(DB_PATH, read_only=True)

    # Inspect exact columns of jobs_technical
    cols = [c[0] for c in con.execute("DESCRIBE jobs_technical").fetchall()]
    print("jobs_technical cols:", len(cols), cols[:30])

    # Check Beijing year breakdown for jobs_technical
    yr_stats = con.execute("""
        SELECT
            year,
            COUNT(*) AS total_tech,
            SUM(CASE WHEN work_city LIKE '%北京%' THEN 1 ELSE 0 END) AS bj_tech,
            SUM(CASE WHEN work_city LIKE '%北京%' AND education_band = 'bachelor' AND explicit_entry THEN 1 ELSE 0 END) AS bj_entry_bach,
            SUM(CASE WHEN work_city LIKE '%北京%' AND education_band IN ('bachelor', 'unspecified') AND (explicit_entry OR unspecified_or_unlimited_experience) THEN 1 ELSE 0 END) AS bj_broad_entry
        FROM jobs_technical
        WHERE year BETWEEN 2018 AND 2026
        GROUP BY year
        ORDER BY year
    """).fetchall()
    print("Year breakdown (year, total_tech, bj_tech, bj_entry_bach, bj_broad_entry):")
    for row in yr_stats:
        print(" ", row)

    # Check whether jobs_technical has skill columns or text columns
    skill_cols = [c for c in cols if any(k in c.lower() for k in ("skill", "desc", "text", "atomic", "coarse", "family", "snippet"))]
    print("Skill/text cols in jobs_technical:", skill_cols[:40])

    # Export strictly separated temporal slices for Beijing technical jobs (de-identified: no raw employer_name)
    # Use hash of employer_key for employer-cluster bootstrap without exposing employer identity
    select_cols = ", ".join([f'"{c}"' for c in cols if c not in ("employer_name", "company_location", "work_location")])
    base_where = "work_city LIKE '%北京%' AND year BETWEEN 2018 AND 2026"

    slices = [
        ("demand_nat_exp_2018_2022.parquet", PROC_DIR, "year BETWEEN 2018 AND 2022"),
        ("demand_train_2018_2023.parquet", PROC_DIR, "year BETWEEN 2018 AND 2023"),
        ("demand_dev_2024.parquet", PROC_DIR, "year = 2024"),
        ("demand_holdout_2025.parquet", BLIND_DIR, "year = 2025"),
        ("demand_drift_2026q1.parquet", BLIND_DIR, "year = 2026"),
    ]

    manifest = {
        "source_db": DB_PATH,
        "year_breakdown": [
            {"year": r[0], "total_tech": r[1], "bj_tech": r[2], "bj_entry_bach": r[3], "bj_broad_entry": r[4]}
            for r in yr_stats
        ],
        "slices": {},
    }

    for fname, folder, cond in slices:
        out_p = folder / fname
        sql = f"""
            COPY (
                SELECT {select_cols}
                FROM jobs_technical
                WHERE {base_where} AND ({cond})
            ) TO '{out_p.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
        con.execute(sql)
        cnt = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_p.as_posix()}')").fetchone()[0]
        manifest["slices"][fname] = {
            "path": str(out_p),
            "blind": folder == BLIND_DIR,
            "rows": cnt,
            "bytes": out_p.stat().st_size,
            "sha256": sha256_file(out_p),
        }
        print(f"Wrote {out_p} ({cnt} rows, {out_p.stat().st_size/1024/1024:.2f} MB)")

    # Also export Beijing enterprise registry summary (by industry & cohort) from registry_04_agg & registry_03
    ent_out = PROC_DIR / "beijing_enterprise_macro.parquet"
    con.execute(f"""
        COPY (
            SELECT
                "成立年份" AS est_year,
                "行业一级" AS industry_l1,
                "登记状态" AS reg_status,
                CAST("主体数" AS BIGINT) AS firm_count,
                TRY_CAST("平均注册资本(万元)" AS DOUBLE) AS avg_capital_wan
            FROM registry_04_agg_prov_year_industry_status
            WHERE "省份" LIKE '%北京%'
        ) TO '{ent_out.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    ent_cnt = con.execute(f"SELECT COUNT(*) FROM read_parquet('{ent_out.as_posix()}')").fetchone()[0]
    manifest["beijing_enterprise_macro"] = {"path": str(ent_out), "rows": ent_cnt, "sha256": sha256_file(ent_out)}

    # Export national technical trend summary (2018-2026) for macro comparison
    nat_out = PROC_DIR / "national_tech_yearly.parquet"
    con.execute(f"""
        COPY (
            SELECT
                year,
                source_archive,
                education_band,
                COUNT(*) AS ads,
                MEDIAN(CASE WHEN salary_valid THEN salary_mid ELSE NULL END) AS median_salary,
                AVG(CASE WHEN salary_valid THEN salary_mid ELSE NULL END) AS mean_salary
            FROM jobs_technical
            WHERE year BETWEEN 2018 AND 2026
            GROUP BY year, source_archive, education_band
            ORDER BY year, source_archive, education_band
        ) TO '{nat_out.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    con.close()

    mpath = PROC_DIR / "demand_slices_manifest.json"
    mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SLICING_DONE:", mpath)


if __name__ == "__main__":
    main()
