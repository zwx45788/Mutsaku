# Mutsaku · 数据支撑材料（EvoAlign-OS）

本材料用于核对方案报告中的全部关键结果。所有报告、PPT、视频与驾驶舱数字取自同一份封印文件
`outputs/verified_master_results.json`（SHA-256 `de9a8115c9cb11f92432052409ecd3f1b9b8333363a23c07169d03c5cde012e1`）。

## 1. 一步复核（约 3 秒，无需求解器、大模型或网络）

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/experiments/verify_key_results.py
```

脚本自动定位仓库根目录（无需 `--root`，可在任意工作目录下调用），读取封印结果、
`outputs/plan_snapshots.json`（各对照组逐方向课程清单）与 `data/verification/`（脱敏逐岗位特征矩阵），
重新计算并逐项比对，输出 `outputs/verification_report.json`。当前结果：**583/583 项一致**。

`outputs/verification_report.json` 是运行产物（含 `runtime_seconds`），因此**不在** `MANIFEST_SHA256.json`
的校验清单内；用 `--out <path>` 可写到别处。

| 复核组 | 内容 |
|---|---|
| 0 封印 | 结果文件 SHA-256、课程快照与结果文件的对应 |
| 1 消融 | 3 专业 × 3 期 × 6 组的达标率、就绪度、最弱方向、L1、OT |
| 2 配对区间 | Full 相对 B0–B4 的雇主聚类配对自助法 95% 区间（600 次，种子 42） |
| 3 自然实验 / 帕累托 | 北工商 2022/2025 版与预测方案、三档预算方案的 2025 回放达标率 |
| 4 任务1 | 11 份方案的 L1 与 OT（Wasserstein-1 精确线性规划） |
| 5 招生调控 | 2018–2024 年度计数 → MPC 稳定性指标；就业报告指标 → 逐年决策 |
| 6 多智能体沙盘 | 300 名学生智能体 × 8 学期推演，3 专业 × 现状/方案 × 三情景的学生匹配度 |
| 7 ReplayBundle | `outputs/replay_bundle/` 中逐代轨迹、记忆库、Harness 基因组与 Qwen 调用原文的 SHA-256 清单与根哈希 |
| 9 2025 滚动复验 | 北京 10 所高校官网公布的 2025 年度增设专业（部分真值）上，EvoAlign 与惯性基线 Top-5 的命中与覆盖 |
| 8 预注册盲测 | 预注册文件哈希、登记早于数据抽取、冻结快照与脚本未改动；津冀/京外两个从未读取人群上的达标率、配对区间与 24 条假设判定 |

## 2. 关键结果的输入样例与输出

| 关键结果 | 最小输入样例 | 对应输出 | 复现方式 |
|---|---|---|---|
| 岗位→技能特征提取 | `data/verification/sample_jd_input.csv`（30 条脱敏岗位） | `sample_jd_output.csv`（岗位族 + 技能清单） | `python examples/extract_skills_demo.py` |
| 课程→技能抽取（Qwen） | `data/processed/university_curricula_enriched.json` 中的课程文本 | 同文件 `extracted_skills` 字段 | `outputs/replay_ledger.jsonl` 按请求哈希可查原始请求/响应 |
| 任务 1 三维差距 | `data/verification/benchmark_pathways.json` + `task1_targets.npz` | 《关键结果核对表》任务1基准 | 复核组 4 |
| 课程增删改与硬约束 | `outputs/plan_snapshots.json` 中 `B0_Freeze` 与 `Full_EvoAlign_OS` | 核对表「课程动作」、结果文件 `constraint_checks` | 复核组 1 |
| 达标率与配对区间 | `features_replay_2025.npz`（2,645 岗位 × 48 技能）+ 课程快照 | 核对表「六组消融」「配对区间」 | 复核组 1–2 |
| 招生调控 | `mpc_yearly_counts.csv`、`employment_feedback_series.json` | 核对表「就业反馈与招生」 | 复核组 5 |

## 3. 完整重跑（可选）

完整实验需要官方基础数据集清洗后的招聘切片（约 100 MB，超出提交体积限制，未随附），
由 `src/ingestion/build_demand_slices.py` 从官方数据生成，放在 `data/processed/` 与 `data/blind_holdout/` 后运行：

```bash
PYTHONPATH=src python src/experiments/run_unified_v2.py      # 约 7 分钟；大模型响应优先从重放账本读取
python src/experiments/generate_figures.py
PYTHONPATH=src python -m uvicorn server.cockpit_server:app --port 8765   # 在线驾驶舱
```

离线驾驶舱：直接打开 `web/index.html`（图表与全部结果可查看，实时重算需启动上述服务）。

驾驶舱各端点在**没有**招聘切片时的行为（已显式处理，不会 500）：

| 端点 | 有切片 | 无切片 |
|---|---|---|
| `GET /api/health`、`GET /api/master`、`GET /`、`/figures/*` | 正常 | 正常 |
| `POST /api/mpc` | 正常 | 正常（合成轨迹基准） |
| `POST /api/whatif` | 正常 | **503** + 中文说明缺哪两个 parquet、如何补、以及离线核验替代命令 |
| `POST /api/whatif`（`program_id` 不存在） | **404** + 可用专业列表 | 同左 |

调用大模型需在本机环境变量中提供自己的千问 API 密钥；本材料不含任何密钥。

## 4. 目录

| 路径 | 内容 |
|---|---|
| `自采数据登记表.xlsx` | 48 份自采公开文件的名称、网址、采集时间、方式、规模、合规结论与 SHA-256 |
| `关键结果核对表.xlsx` | 消融、配对区间、任务1、课程动作、风险矩阵、就业反馈、独立复核结果 |
| `数据合规自查声明.md` | 合规自查声明（待队长签字） |
| `outputs/` | 封印结果、课程快照、复核报告、大模型重放账本、隔离数据访问审计、图表 |
| `data/verification/` | 脱敏复核输入 |
| `data/processed/` | 知识图谱、培养方案课程、专业备案、紧缺目录、就业报告指标等结构化数据 |
| `src/` | 采集、本体、差距引擎、求解器、仿真、自进化、实验、复核与交付生成代码。正式实验入口为 `src/experiments/run_unified_v2.py`；`run_m3_experiments.py`、`run_structural_repair.py` 为早期版本，仅保留作对照，不产生报告数字 |
| `web/` | 决策驾驶舱 |
| `examples/` | 最小输入→输出演示脚本 |
