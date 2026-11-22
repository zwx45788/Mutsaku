"""Build EvoAlign-OS Dual-Layer Dynamic Industry-Education Knowledge Graph (M1).
Combines:
  - 48 Canonical Atomic Skills anchored to CS2023 KA, MOE 101 Plan, ESCO codes, Bloom levels & BJ 2025/2026 Shortage Catalog
  - Prerequisite DAG & Major/Course Lifecycle State Machine (Emerging->Incubating->Core->Declining->TeachOut->Sunset)
  - Real Qwen3.8-flash JSON-Schema extraction of (skill_id, bloom_level 1..6, weight) over unique Beijing university courses
    enriched with real BJUT/BISTU full syllabus text
  - Formal extraction reliability check (cross-pass agreement on 24 stratified courses)
  - Local BGE-small-zh-v1.5 semantic ground-distance matrix C_{ij} = 1 - cos(e_i, e_j) for Optimal Transport
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

sys.path.insert(0, "evoalign_os/src")
from llm.client import QwenClient

PROC_DIR = Path("evoalign_os/data/processed")
BGE_MODEL_DIRS = list(Path("work/modern_data/models").glob("**/*bge*"))

# 48 Canonical Atomic Skills across 8 Domains (anchored to CS2023, 101 Plan, ESCO)
CANONICAL_SKILLS: List[Dict[str, Any]] = [
    # 1. Math & Theoretical Foundations (Long half-life, CS2023 Core)
    {"id": "discrete_math", "zh": "离散数学与数理逻辑", "en": "Discrete Mathematics & Logic", "domain": "foundations", "cs2023_ka": "MSF", "esco_legacy_code": "S1.1.1", "core": True, "half_life": 25.0, "stage": "core", "job_bloom": 3, "synonyms": ["离散数学", "图论", "集合论", "数理逻辑", "组合数学"]},
    {"id": "linear_algebra_opt", "zh": "线性代数与最优化", "en": "Linear Algebra & Optimization", "domain": "foundations", "cs2023_ka": "MSF", "esco_legacy_code": "S1.1.2", "core": True, "half_life": 25.0, "stage": "core", "job_bloom": 4, "synonyms": ["线性代数", "矩阵", "最优化", "凸优化", "数值分析", "运筹学"]},
    {"id": "prob_stats", "zh": "概率统计与随机建模", "en": "Probability & Statistical Modeling", "domain": "foundations", "cs2023_ka": "MSF", "esco_legacy_code": "S1.1.3", "core": True, "half_life": 25.0, "stage": "core", "job_bloom": 4, "synonyms": ["概率论", "数理统计", "统计学", "贝叶斯", "假设检验", "A/B测试", "随机过程"]},
    {"id": "dsa", "zh": "数据结构与算法设计", "en": "Data Structures & Algorithms", "domain": "foundations", "cs2023_ka": "AL", "esco_legacy_code": "S5.1.1", "core": True, "half_life": 20.0, "stage": "core", "job_bloom": 5, "synonyms": ["数据结构", "算法", "动态规划", "图算法", "刷题", "leetcode", "复杂度分析"]},

    # 2. Computer Systems & Architecture (CS2023 Core / 101 Plan)
    {"id": "comp_arch", "zh": "计算机组成与体系结构", "en": "Computer Organization & Architecture", "domain": "systems", "cs2023_ka": "AR", "esco_legacy_code": "S5.2.1", "core": True, "half_life": 15.0, "stage": "core", "job_bloom": 3, "synonyms": ["组成原理", "体系结构", "CPU", "数字逻辑", "指令集", "RISC-V", "ARM", "x86", "汇编"]},
    {"id": "os_kernel", "zh": "操作系统与内核机制", "en": "Operating Systems & Concurrency", "domain": "systems", "cs2023_ka": "OS", "esco_legacy_code": "S5.2.2", "core": True, "half_life": 15.0, "stage": "core", "job_bloom": 4, "synonyms": ["操作系统", "内核", "进程", "线程", "内存管理", "文件系统", "并发", "锁", "OS"]},
    {"id": "linux_shell", "zh": "Linux系统与Shell运维", "en": "Linux Administration & Shell Scripting", "domain": "systems", "cs2023_ka": "OS", "esco_legacy_code": "S5.2.3", "core": False, "half_life": 10.0, "stage": "core", "job_bloom": 4, "synonyms": ["linux", "unix", "shell", "bash", "centos", "ubuntu", "运维", "systemd"]},
    {"id": "comp_net", "zh": "计算机网络与通信协议", "en": "Computer Networks & Protocols", "domain": "systems", "cs2023_ka": "NC", "esco_legacy_code": "S5.3.1", "core": True, "half_life": 15.0, "stage": "core", "job_bloom": 4, "synonyms": ["计算机网络", "tcp", "udp", "http", "https", "socket", "网络编程", "grpc", "dns", "cdn"]},
    {"id": "compiler_pl", "zh": "编译原理与编程语言理论", "en": "Compilers & Programming Languages", "domain": "systems", "cs2023_ka": "FPL", "esco_legacy_code": "S5.1.2", "core": True, "half_life": 15.0, "stage": "core", "job_bloom": 3, "synonyms": ["编译原理", "编译器", "llvm", "mlir", "词法分析", "语法分析", "虚拟机", "jit"]},
    {"id": "parallel_gpu", "zh": "并行计算与GPU/CUDA算力优化", "en": "Parallel Computing & GPU/CUDA Optimization", "domain": "systems", "cs2023_ka": "PDC", "esco_legacy_code": "S5.2.4", "core": False, "half_life": 6.0, "stage": "emerging", "job_bloom": 5, "synonyms": ["cuda", "gpu", "并行计算", "openmp", "mpi", "triton", "算子优化", "异构计算", "tensorrt", "hpc"]},

    # 3. Programming Languages & Core Software Engineering
    {"id": "c_cpp", "zh": "C/C++系统级编程", "en": "C/C++ Systems Programming", "domain": "software_eng", "cs2023_ka": "SDF", "esco_legacy_code": "S5.1.3", "core": True, "half_life": 15.0, "stage": "core", "job_bloom": 5, "synonyms": ["c语言", "c++", "cpp", "stl", "cmake", "指针", "内存优化", "高性能c++"]},
    {"id": "python_eng", "zh": "Python工程与科学计算", "en": "Python Engineering & Scientific Computing", "domain": "software_eng", "cs2023_ka": "SDF", "esco_legacy_code": "S5.1.4", "core": True, "half_life": 10.0, "stage": "core", "job_bloom": 5, "synonyms": ["python", "numpy", "pandas", "fastapi", "django", "flask", "asyncio", "pytest"]},
    {"id": "java_spring", "zh": "Java后端与企业级架构", "en": "Java Backend & Spring Ecosystem", "domain": "software_eng", "cs2023_ka": "SE", "esco_legacy_code": "S5.1.5", "core": False, "half_life": 8.0, "stage": "mature", "job_bloom": 5, "synonyms": ["java", "jvm", "spring", "springboot", "springcloud", "mybatis", "面向对象", "j2ee"]},
    {"id": "golang_rust", "zh": "Go/Rust现代并发与系统开发", "en": "Go & Rust Modern Systems Development", "domain": "software_eng", "cs2023_ka": "FPL", "esco_legacy_code": "S5.1.6", "core": False, "half_life": 7.0, "stage": "growing", "job_bloom": 5, "synonyms": ["golang", "go语言", "rust", "gin", "tokio", "内存安全", "协程"]},
    {"id": "web_frontend", "zh": "Web前端与交互工程", "en": "Web Frontend & Interactive Engineering", "domain": "software_eng", "cs2023_ka": "HCI", "esco_legacy_code": "S5.4.1", "core": False, "half_life": 5.0, "stage": "mature", "job_bloom": 4, "synonyms": ["javascript", "typescript", "html", "css", "vue", "react", "node.js", "前端", "web开发", "小程序"]},
    {"id": "mobile_harmony", "zh": "移动应用与鸿蒙生态开发", "en": "Mobile & HarmonyOS Application Dev", "domain": "software_eng", "cs2023_ka": "SPD", "esco_legacy_code": "S5.4.2", "core": False, "half_life": 5.0, "stage": "growing", "job_bloom": 4, "synonyms": ["android", "ios", "swift", "kotlin", "flutter", "鸿蒙", "harmonyos", "移动应用", "app开发"]},
    {"id": "soft_eng_arch", "zh": "软件工程、敏捷与架构设计", "en": "Software Engineering & System Architecture", "domain": "software_eng", "cs2023_ka": "SE", "esco_legacy_code": "S5.5.1", "core": True, "half_life": 12.0, "stage": "core", "job_bloom": 5, "synonyms": ["软件工程", "设计模式", "uml", "架构设计", "微服务", "git", "代码规范", "需求分析", "领域驱动"]},
    {"id": "qa_testing", "zh": "软件测试、自动化QA与质量保障", "en": "Software Testing & Automated QA", "domain": "software_eng", "cs2023_ka": "SE", "esco_legacy_code": "S5.5.2", "core": False, "half_life": 8.0, "stage": "core", "job_bloom": 4, "synonyms": ["软件测试", "自动化测试", "单元测试", "性能测试", "selenium", "jmeter", "pytest", "qa", "质量保障"]},
    {"id": "ai_native_se", "zh": "AI原生软件工程与代码智能体评测", "en": "AI-Native Software Engineering & Eval", "domain": "software_eng", "cs2023_ka": "SE", "esco_legacy_code": "S5.5.3", "core": False, "half_life": 3.0, "stage": "emerging", "job_bloom": 5, "synonyms": ["ai编程", "copilot", "cursor", "代码生成", "ai辅助开发", "软件评测", "harness", "vibe coding"]},

    # 4. Cloud Native, Distributed Systems & DevOps
    {"id": "dist_systems", "zh": "分布式系统与高并发架构", "en": "Distributed Systems & High-Concurrency", "domain": "cloud_dist", "cs2023_ka": "PDC", "esco_legacy_code": "S5.3.2", "core": False, "half_life": 10.0, "stage": "growing", "job_bloom": 5, "synonyms": ["分布式", "高并发", "kafka", "rabbitmq", "消息队列", "zookeeper", "raft", "一致性", "rpc", "缓存"]},
    {"id": "cloud_k8s", "zh": "云计算、容器化与Kubernetes", "en": "Cloud Computing, Docker & Kubernetes", "domain": "cloud_dist", "cs2023_ka": "SF", "esco_legacy_code": "S5.3.3", "core": False, "half_life": 6.0, "stage": "growing", "job_bloom": 5, "synonyms": ["云计算", "docker", "kubernetes", "k8s", "云原生", "aws", "阿里云", "容器", "serverless", "istio"]},
    {"id": "devops_sre", "zh": "DevOps、CI/CD与可观测性SRE", "en": "DevOps, CI/CD & SRE Observability", "domain": "cloud_dist", "cs2023_ka": "SE", "esco_legacy_code": "S5.3.4", "core": False, "half_life": 6.0, "stage": "growing", "job_bloom": 4, "synonyms": ["devops", "ci/cd", "jenkins", "gitlab", "prometheus", "grafana", "sre", "自动化部署", "监控", "可观测"]},

    # 5. Data Engineering, Databases & Big Data
    {"id": "sql_rdbms", "zh": "关系数据库与SQL调优", "en": "Relational Databases & SQL Tuning", "domain": "data_db", "cs2023_ka": "DM", "esco_legacy_code": "S5.6.1", "core": True, "half_life": 15.0, "stage": "core", "job_bloom": 5, "synonyms": ["数据库", "sql", "mysql", "postgresql", "oracle", "索引", "事务", "数据库原理"]},
    {"id": "nosql_vector_db", "zh": "NoSQL、缓存与向量数据库", "en": "NoSQL, Caching & Vector Databases", "domain": "data_db", "cs2023_ka": "DM", "esco_legacy_code": "S5.6.2", "core": False, "half_life": 6.0, "stage": "growing", "job_bloom": 4, "synonyms": ["redis", "mongodb", "elasticsearch", "milvus", "向量数据库", "faiss", "图数据库", "neo4j", "nosql"]},
    {"id": "big_data_eng", "zh": "大数据平台与实时数仓工程", "en": "Big Data Pipelines & Lakehouse Engineering", "domain": "data_db", "cs2023_ka": "DM", "esco_legacy_code": "S5.6.3", "core": False, "half_life": 6.0, "stage": "mature", "job_bloom": 5, "synonyms": ["hadoop", "spark", "flink", "hive", "数据仓库", "etl", "大数据", "clickhouse", "doris", "数仓"]},
    {"id": "data_mining_bi", "zh": "数据分析、可视化与商业智能", "en": "Data Analytics, Visualization & BI", "domain": "data_db", "cs2023_ka": "DM", "esco_legacy_code": "S5.6.4", "core": False, "half_life": 8.0, "stage": "core", "job_bloom": 4, "synonyms": ["数据分析", "数据挖掘", "可视化", "tableau", "powerbi", "商业智能", "bi", "报表", "金融数据"]},

    # 6. AI, Machine Learning, LLM & Agent Systems
    {"id": "ml_classic", "zh": "机器学习与统计学习", "en": "Machine Learning & Statistical Learning", "domain": "ai_llm", "cs2023_ka": "AI", "esco_legacy_code": "S5.7.1", "core": True, "half_life": 10.0, "stage": "core", "job_bloom": 5, "synonyms": ["机器学习", "sklearn", "xgboost", "lightgbm", "特征工程", "分类", "回归", "聚类", "模式识别"]},
    {"id": "deep_learning", "zh": "深度学习与PyTorch神经网络", "en": "Deep Learning & PyTorch Architectures", "domain": "ai_llm", "cs2023_ka": "AI", "esco_legacy_code": "S5.7.2", "core": False, "half_life": 6.0, "stage": "growing", "job_bloom": 5, "synonyms": ["深度学习", "pytorch", "tensorflow", "神经网络", "cnn", "transformer", "反向传播", "模型训练"]},
    {"id": "nlp_text", "zh": "自然语言处理与信息检索", "en": "NLP & Information Retrieval", "domain": "ai_llm", "cs2023_ka": "AI", "esco_legacy_code": "S5.7.3", "core": False, "half_life": 5.0, "stage": "growing", "job_bloom": 5, "synonyms": ["自然语言处理", "nlp", "文本挖掘", "信息检索", "知识图谱", "bert", "语义理解", "词向量"]},
    {"id": "cv_multimodal", "zh": "计算机视觉与多模态理解", "en": "Computer Vision & Multimodal AI", "domain": "ai_llm", "cs2023_ka": "AI", "esco_legacy_code": "S5.7.4", "core": False, "half_life": 5.0, "stage": "growing", "job_bloom": 5, "synonyms": ["计算机视觉", "opencv", "yolo", "目标检测", "图像识别", "图像处理", "多模态", "vlm", "三维视觉", "slam"]},
    {"id": "llm_posttrain", "zh": "大语言模型微调、对齐与推理部署", "en": "LLM Fine-tuning, Alignment & Serving", "domain": "ai_llm", "cs2023_ka": "AI", "esco_legacy_code": "S5.7.5", "core": False, "half_life": 3.0, "stage": "emerging", "job_bloom": 5, "synonyms": ["大模型", "llm", "lora", "sft", "rlhf", "dpo", "vllm", "deepspeed", "量化", "推理加速", "aigc", "生成式ai"]},
    {"id": "agent_rag", "zh": "AI智能体、RAG与工具编排系统", "en": "AI Agents, RAG & Tool Orchestration", "domain": "ai_llm", "cs2023_ka": "AI", "esco_legacy_code": "S5.7.6", "core": False, "half_life": 3.0, "stage": "emerging", "job_bloom": 6, "synonyms": ["agent", "智能体", "rag", "langchain", "langgraph", "mcp", "prompt", "提示词", "工作流", "多智能体", "函数调用"]},
    {"id": "rl_decision", "zh": "强化学习与序贯决策优化", "en": "Reinforcement Learning & Sequential Decision", "domain": "ai_llm", "cs2023_ka": "AI", "esco_legacy_code": "S5.7.7", "core": False, "half_life": 6.0, "stage": "growing", "job_bloom": 5, "synonyms": ["强化学习", "ppo", "grpo", "马尔可夫", "博弈论", "决策规划", "推荐系统", "搜索广告"]},
    {"id": "mlops_eval", "zh": "MLOps、数据闭环与模型评测治理", "en": "MLOps, Data Flywheel & Model Evaluation", "domain": "ai_llm", "cs2023_ka": "SE", "esco_legacy_code": "S5.7.8", "core": False, "half_life": 4.0, "stage": "emerging", "job_bloom": 5, "synonyms": ["mlops", "llmops", "模型部署", "评测集", "benchmark", "模型监控", "特征平台", "数据闭环"]},

    # 7. Embedded, IoT, Robotics & Embodied AI
    {"id": "embedded_mcu", "zh": "嵌入式系统、RTOS与硬件接口", "en": "Embedded Systems, RTOS & MCU", "domain": "embodied_iot", "cs2023_ka": "AR", "esco_legacy_code": "S5.8.1", "core": False, "half_life": 10.0, "stage": "core", "job_bloom": 5, "synonyms": ["嵌入式", "单片机", "stm32", "rtos", "freertos", "驱动开发", "fpga", "verilog", "dsp", "接口技术"]},
    {"id": "iot_edge", "zh": "物联网感知与边缘智能计算", "en": "IoT Sensing & Edge AI Computing", "domain": "embodied_iot", "cs2023_ka": "NC", "esco_legacy_code": "S5.8.2", "core": False, "half_life": 6.0, "stage": "mature", "job_bloom": 4, "synonyms": ["物联网", "iot", "mqtt", "传感器", "边缘计算", "工业互联网", "rfid", "无线传感"]},
    {"id": "robotics_embodied", "zh": "机器人控制、ROS与具身智能", "en": "Robotics, ROS & Embodied AI", "domain": "embodied_iot", "cs2023_ka": "AI", "esco_legacy_code": "S5.8.3", "core": False, "half_life": 5.0, "stage": "emerging", "job_bloom": 5, "synonyms": ["机器人", "ros", "ros2", "具身智能", "运动控制", "自动驾驶", "路径规划", "仿真", "isaac", "vla"]},
    {"id": "control_signal", "zh": "自动控制原理与数字信号处理", "en": "Control Theory & Digital Signal Processing", "domain": "embodied_iot", "cs2023_ka": "MSF", "esco_legacy_code": "S5.8.4", "core": False, "half_life": 15.0, "stage": "core", "job_bloom": 4, "synonyms": ["自动控制", "控制原理", "信号处理", "plc", "电机控制", "现代控制", "滤波", "语音信号"]},

    # 8. Security, Graphics, Legacy & Cross-Domain
    {"id": "cyber_sec", "zh": "网络安全、密码学与攻防实践", "en": "Cybersecurity, Cryptography & Defense", "domain": "security_domain", "cs2023_ka": "SEC", "esco_legacy_code": "S5.9.1", "core": True, "half_life": 10.0, "stage": "growing", "job_bloom": 5, "synonyms": ["网络安全", "信息安全", "密码学", "渗透测试", "漏洞", "加密", "等保", "安全开发", "零信任"]},
    {"id": "ai_safety_privacy", "zh": "AI安全、隐私计算与数据合规治理", "en": "AI Safety, Privacy Computing & Governance", "domain": "security_domain", "cs2023_ka": "SEP", "esco_legacy_code": "S5.9.2", "core": False, "half_life": 5.0, "stage": "emerging", "job_bloom": 4, "synonyms": ["ai安全", "隐私计算", "联邦学习", "数据安全", "数据合规", "大模型安全", "内容安全", "伦理"]},
    {"id": "graphics_game", "zh": "计算机图形学、渲染与交互仿真", "en": "Computer Graphics, Rendering & Game Dev", "domain": "security_domain", "cs2023_ka": "GIT", "esco_legacy_code": "S5.9.3", "core": False, "half_life": 8.0, "stage": "mature", "job_bloom": 4, "synonyms": ["图形学", "opengl", "vulkan", "unity", "unreal", "游戏开发", "3d", "动画", "数字孪生", "vr", "ar"]},
    {"id": "blockchain_fintech", "zh": "金融科技、量化与区块链应用", "en": "FinTech, Quantitative Systems & Blockchain", "domain": "security_domain", "cs2023_ka": "SPD", "esco_legacy_code": "S5.9.4", "core": False, "half_life": 6.0, "stage": "mature", "job_bloom": 4, "synonyms": ["金融科技", "量化", "风控", "区块链", "智能合约", "支付系统", "金融信息"]},
    {"id": "hci_product", "zh": "人机交互、产品设计与工程管理", "en": "HCI, Product Design & Engineering Mgmt", "domain": "security_domain", "cs2023_ka": "HCI", "esco_legacy_code": "S5.9.5", "core": False, "half_life": 10.0, "stage": "core", "job_bloom": 4, "synonyms": ["人机交互", "hci", "产品经理", "项目管理", "原型设计", "用户体验", "ux", "it创业"]},
    {"id": "tech_comm_ethics", "zh": "工程伦理、科技写作与团队协作", "en": "Engineering Ethics & Technical Communication", "domain": "security_domain", "cs2023_ka": "SEP", "esco_legacy_code": "S5.9.6", "core": True, "half_life": 20.0, "stage": "core", "job_bloom": 3, "synonyms": ["工程伦理", "科技写作", "论文写作", "职业素养", "团队协作", "开源社区"]},
    {"id": "legacy_erp_vb", "zh": "传统桌面/旧式数据库与遗留系统(衰退类)", "en": "Legacy Desktop/Oracle Forms/Older Stack", "domain": "security_domain", "cs2023_ka": "SPD", "esco_legacy_code": "S5.9.9", "core": False, "half_life": 3.0, "stage": "declining", "job_bloom": 3, "synonyms": ["vb", "visual basic", "jsp", "struts", "oracle管理", "xml", "flash", "winform", "mfc", "delphi", "php旧架构", "多媒体技术"]},
    {"id": "open_source_collab", "zh": "开源工程、Linux内核贡献与工程实践", "en": "Open-Source Engineering & Collaborative Dev", "domain": "software_eng", "cs2023_ka": "SE", "esco_legacy_code": "S5.5.4", "core": False, "half_life": 8.0, "stage": "growing", "job_bloom": 5, "synonyms": ["开源", "github", "开源操作系统", "开源软件", "pull request", "代码评审"]},
    {"id": "eda_ic_design", "zh": "集成电路、EDA与芯片软件栈", "en": "IC Design, EDA & Chip Software Stack", "domain": "embodied_iot", "cs2023_ka": "AR", "esco_legacy_code": "S5.8.5", "core": False, "half_life": 8.0, "stage": "growing", "job_bloom": 5, "synonyms": ["集成电路", "芯片", "eda", "soc", "ic设计", "数字电路", "模拟电路", "硬件描述"]},
    {"id": "bio_sci_computing", "zh": "科学计算、AI4Science与交叉应用", "en": "Scientific Computing & AI4Science", "domain": "ai_llm", "cs2023_ka": "MSF", "esco_legacy_code": "S5.7.9", "core": False, "half_life": 8.0, "stage": "emerging", "job_bloom": 4, "synonyms": ["科学计算", "ai4science", "计算生物", "食品智能", "智慧交通", "物理仿真", "数字经济"]},
]

SKILL_BY_ID = {s["id"]: s for s in CANONICAL_SKILLS}

# Directed Prerequisite DAG between skills (u -> v means u is prerequisite for v)
SKILL_PREREQ_EDGES: List[Tuple[str, str]] = [
    ("discrete_math", "dsa"),
    ("c_cpp", "dsa"),
    ("c_cpp", "comp_arch"),
    ("comp_arch", "os_kernel"),
    ("dsa", "os_kernel"),
    ("dsa", "compiler_pl"),
    ("dsa", "sql_rdbms"),
    ("os_kernel", "dist_systems"),
    ("comp_net", "dist_systems"),
    ("linux_shell", "cloud_k8s"),
    ("cloud_k8s", "devops_sre"),
    ("comp_arch", "parallel_gpu"),
    ("os_kernel", "parallel_gpu"),
    ("linear_algebra_opt", "ml_classic"),
    ("prob_stats", "ml_classic"),
    ("python_eng", "ml_classic"),
    ("ml_classic", "deep_learning"),
    ("deep_learning", "nlp_text"),
    ("deep_learning", "cv_multimodal"),
    ("deep_learning", "llm_posttrain"),
    ("nlp_text", "llm_posttrain"),
    ("llm_posttrain", "agent_rag"),
    ("nosql_vector_db", "agent_rag"),
    ("python_eng", "agent_rag"),
    ("ml_classic", "rl_decision"),
    ("deep_learning", "mlops_eval"),
    ("cloud_k8s", "mlops_eval"),
    ("soft_eng_arch", "ai_native_se"),
    ("agent_rag", "ai_native_se"),
    ("comp_arch", "embedded_mcu"),
    ("embedded_mcu", "robotics_embodied"),
    ("control_signal", "robotics_embodied"),
    ("cv_multimodal", "robotics_embodied"),
    ("comp_net", "cyber_sec"),
    ("os_kernel", "cyber_sec"),
    ("cyber_sec", "ai_safety_privacy"),
    ("llm_posttrain", "ai_safety_privacy"),
    ("sql_rdbms", "big_data_eng"),
    ("dist_systems", "big_data_eng"),
]


def compute_bj_shortage_weights() -> Dict[str, float]:
    """Derive official Beijing shortage calibration weights directly from parsed BJ 2025/2026 talent catalogs."""
    p = PROC_DIR / "beijing_talent_shortage_catalog.json"
    weights = {s["id"]: 1.0 for s in CANONICAL_SKILLS}
    if not p.exists():
        return weights
    cat = json.loads(p.read_text(encoding="utf-8"))
    entries = cat.get("entries", [])
    hit_stars: Dict[str, List[int]] = defaultdict(list)
    for e in entries:
        txt = " ".join(e.get("raw_cells", [])).lower()
        stars = int(e.get("stars", 3))
        for s in CANONICAL_SKILLS:
            if any(syn.lower() in txt for syn in s["synonyms"]) or s["zh"][:4].lower() in txt:
                hit_stars[s["id"]].append(stars)
    for sid, st_list in hit_stars.items():
        avg_star = sum(st_list) / len(st_list)
        # Bounded calibration multiplier in [1.0, 1.45] reflecting Beijing official star rating & frequency
        boost = min(0.45, 0.08 * math.log1p(len(st_list)) + 0.06 * max(0.0, avg_star - 2.0))
        weights[sid] = round(1.0 + boost, 4)
    return weights


def compute_semantic_ground_distance() -> Tuple[List[str], List[List[float]], str]:
    """Compute 48x48 semantic ground distance matrix C_ij = 1 - cos(e_i, e_j) using local BGE-small-zh-v1.5."""
    skill_ids = [s["id"] for s in CANONICAL_SKILLS]
    texts = [f"{s['zh']} ({s['en']})：{'、'.join(s['synonyms'][:6])}" for s in CANONICAL_SKILLS]

    model_dir = None
    for d in BGE_MODEL_DIRS:
        if (d / "config.json").exists():
            model_dir = d
            break
    # Also check snapshots subfolder under huggingface cache in work/modern_data/models
    if model_dir is None:
        for cfg in Path("work/modern_data/models").rglob("config.json"):
            if "bge" in str(cfg).lower():
                model_dir = cfg.parent
                break

    if model_dir is not None:
        from sentence_transformers import SentenceTransformer
        mdl = SentenceTransformer(str(model_dir), device="cpu")
        emb = mdl.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        sim = np.clip(emb @ emb.T, -1.0, 1.0)
        dist = np.clip(1.0 - sim, 0.0, 2.0)
        np.fill_diagonal(dist, 0.0)
        return skill_ids, np.round(dist, 5).tolist(), f"local_bge:{model_dir}"
    else:
        # Fallback: domain-aware distance
        n = len(skill_ids)
        dist = np.ones((n, n)) * 0.75
        for i in range(n):
            dist[i, i] = 0.0
            for j in range(n):
                if i != j and CANONICAL_SKILLS[i]["domain"] == CANONICAL_SKILLS[j]["domain"]:
                    dist[i, j] = 0.35
        return skill_ids, np.round(dist, 5).tolist(), "domain_fallback"


def normalize_course_key(name: str) -> str:
    s = re.sub(r"[\s（）\(\)ⅠⅡⅢⅣ1234ABCab一二三四上下\-—_]", "", name)
    return s.strip()


def extract_courses_with_qwen(unique_courses: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Batch-extract (skill_id, bloom_level 1..6, weight) via Qwen3.8-flash with strict JSON schema."""
    client = QwenClient(model="qwen3.8-flash", timeout=90.0)
    skill_menu = "\n".join([f"- {s['id']}: {s['zh']} ({s['en']})" for s in CANONICAL_SKILLS])

    system_prompt = (
        "你是高校计算机/人工智能教育与劳动力市场对齐专家。请根据给定高校课程名称、类别及（若有）教学大纲摘要，"
        "将其精准映射到给定48个标准技能ID，并评估该课程对每个技能培养的Bloom认知深度"
        "(1=记忆, 2=理解, 3=应用, 4=分析, 5=评价/系统实现, 6=创造/前沿研发)与支撑权重(weight之和约等于1.0)。\n"
        "严格输出JSON对象，格式为：{\"courses\": [{\"key\": \"...\", \"skills\": [{\"id\": \"skill_id\", \"bloom\": 3, \"weight\": 0.5}], \"is_core_foundation\": true/false}]}\n"
        "每门课选1至4个最直接相关的skill_id，只能使用下方词表中的id，严禁编造新id：\n" + skill_menu
    )

    batches = [unique_courses[i : i + 16] for i in range(0, len(unique_courses), 16)]

    def run_batch(b_idx: int, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        lines = []
        for item in batch:
            syl = f" | 大纲摘要: {item['syllabus_snippet'][:220]}" if item.get("syllabus_snippet") else ""
            lines.append(
                f"- key={item['key']} | 课名={item['course_name']} | 类别={item['category']} | 学分={item['credits']} | 实践学时={item['lab_hours']}{syl}"
            )
        user_prompt = "请为以下课程输出标准JSON技能与Bloom深度映射：\n" + "\n".join(lines)
        res = client.chat_json(
            system=system_prompt,
            user=user_prompt,
            temperature=0.0,
            max_tokens=4096,
            tag=f"m1_course_extract_b{b_idx:02d}",
            enable_thinking=False,
        )
        out = {}
        for row in res["data"].get("courses", []):
            k = str(row.get("key", ""))
            clean_skills = []
            for sk in row.get("skills", []):
                sid = sk.get("id")
                if sid in SKILL_BY_ID:
                    bl = max(1, min(6, int(round(float(sk.get("bloom", 3))))))
                    wt = max(0.05, min(1.0, float(sk.get("weight", 0.3))))
                    clean_skills.append({"id": sid, "bloom": bl, "weight": wt})
            if clean_skills:
                s_wt = sum(x["weight"] for x in clean_skills)
                for x in clean_skills:
                    x["weight"] = round(x["weight"] / s_wt, 4)
            out[k] = {
                "skills": clean_skills,
                "is_core_foundation": bool(row.get("is_core_foundation", False)),
            }
        return out

    extracted: Dict[str, Any] = {}
    done_b = 0
    with ThreadPoolExecutor(max_workers=10) as pool:
        futs = {pool.submit(run_batch, idx, b): idx for idx, b in enumerate(batches)}
        for fut in as_completed(futs):
            extracted.update(fut.result())
            done_b += 1
            if done_b % 10 == 0 or done_b == len(batches):
                print(f"  Qwen course extraction progress: {done_b}/{len(batches)} batches ({len(extracted)} courses)", flush=True)

    # Formal Extraction Reliability Check (arXiv:2606.01982):
    # Re-extract 24 stratified courses with an independent model (qwen3.8-max) & measure Jaccard + Bloom MAE!
    sample_24 = [c for c in unique_courses if c.get("syllabus_snippet")][:12] + [c for c in unique_courses if not c.get("syllabus_snippet")][:12]
    sample_24 = sample_24[:24]
    client_max = QwenClient(model="qwen3.8-max", timeout=120.0)
    s_lines = [
        f"- key={item['key']} | 课名={item['course_name']} | 类别={item['category']} | 学分={item['credits']}"
        + (f" | 大纲摘要: {item['syllabus_snippet'][:200]}" if item.get("syllabus_snippet") else "")
        for item in sample_24
    ]
    res_aud = client_max.chat_json(
        system=system_prompt,
        user="请独立复核并输出以下24门课程的技能与Bloom映射JSON：\n" + "\n".join(s_lines),
        temperature=0.0,
        max_tokens=4096,
        tag="m1_reliability_audit_qwen38max",
        enable_thinking=False,
    )
    aud_map = {}
    for row in res_aud["data"].get("courses", []):
        k = str(row.get("key", ""))
        aud_map[k] = [
            {"id": sk["id"], "bloom": max(1, min(6, int(round(float(sk.get("bloom", 3))))))}
            for sk in row.get("skills", [])
            if sk.get("id") in SKILL_BY_ID
        ]

    jaccards, top1_hits, bloom_diffs = [], [], []
    for item in sample_24:
        k = item["key"]
        s1 = extracted.get(k, {}).get("skills", [])
        s2 = aud_map.get(k, [])
        if not s1 or not s2:
            continue
        set1 = {x["id"] for x in s1}
        set2 = {x["id"] for x in s2}
        jaccards.append(len(set1 & set2) / max(1, len(set1 | set2)))
        top1_hits.append(1.0 if s1[0]["id"] in set2 else 0.0)
        b1 = {x["id"]: x["bloom"] for x in s1}
        b2 = {x["id"]: x["bloom"] for x in s2}
        for common in (set1 & set2):
            bloom_diffs.append(abs(b1[common] - b2[common]))

    reliability = {
        "auditor_model": "qwen3.8-max",
        "extractor_model": "qwen3.8-flash",
        "audited_courses": len(jaccards),
        "mean_skill_jaccard": round(float(np.mean(jaccards)), 4) if jaccards else 0.0,
        "top1_skill_agreement": round(float(np.mean(top1_hits)), 4) if top1_hits else 0.0,
        "bloom_depth_mae": round(float(np.mean(bloom_diffs)), 4) if bloom_diffs else 0.0,
        "token_stats_flash": client.ledger.stats(),
    }
    return extracted, reliability


def main() -> None:
    courses = json.loads((PROC_DIR / "university_curricula_courses.json").read_text(encoding="utf-8"))
    syllabi = json.loads((PROC_DIR / "course_syllabi_corpus.json").read_text(encoding="utf-8"))
    programs = json.loads((PROC_DIR / "university_programs_summary.json").read_text(encoding="utf-8"))

    # Index syllabi by normalized course name
    syl_index: Dict[str, str] = {}
    for s in syllabi:
        k = normalize_course_key(s["course_name"])
        snippet = (s.get("objectives_snippet", "") + " " + s.get("content_snippet", "")).strip()
        if k and (k not in syl_index or len(snippet) > len(syl_index[k])):
            syl_index[k] = snippet

    # Collect unique technical courses (excluding pure general_edu)
    uniq_map: Dict[str, Dict[str, Any]] = {}
    for c in courses:
        if c["category"] == "general_edu":
            continue
        k = normalize_course_key(c["course_name"])
        if not k or len(k) < 2:
            continue
        if k not in uniq_map:
            # Find syllabus match (exact or substring)
            syl = syl_index.get(k, "")
            if not syl:
                for sk, sv in syl_index.items():
                    if len(sk) >= 3 and (sk in k or k in sk):
                        syl = sv
                        break
            uniq_map[k] = {
                "key": k,
                "course_name": c["course_name"],
                "category": c["category"],
                "credits": c["credits"],
                "lab_hours": c["lab_hours"],
                "syllabus_snippet": syl,
            }

    unique_list = sorted(uniq_map.values(), key=lambda x: x["key"])
    syl_matched = sum(1 for x in unique_list if x["syllabus_snippet"])
    print(f"Unique technical courses to extract: {len(unique_list)} ({syl_matched} matched to full official syllabus text)")

    # 1. Official Beijing shortage calibration weights
    bj_weights = compute_bj_shortage_weights()
    for s in CANONICAL_SKILLS:
        s["bj_shortage_weight"] = bj_weights[s["id"]]

    # 2. Local BGE semantic ground distance matrix
    skill_ids, dist_matrix, emb_src = compute_semantic_ground_distance()
    print(f"Computed {len(skill_ids)}x{len(skill_ids)} semantic ground distance matrix via {emb_src}")

    # 3. Real Qwen Schema Extraction + Dual-Model Reliability Audit
    extracted_map, reliability = extract_courses_with_qwen(unique_list)
    print("Extraction reliability:", json.dumps(reliability, ensure_ascii=False))

    # Attach extracted skill*Bloom profiles back to every course
    mapped_cnt = 0
    for c in courses:
        if c["category"] == "general_edu":
            c["extracted_skills"] = []
            c["has_syllabus"] = False
            continue
        k = normalize_course_key(c["course_name"])
        info = extracted_map.get(k, {})
        c["extracted_skills"] = info.get("skills", [])
        c["is_core_foundation"] = info.get("is_core_foundation", False)
        c["has_syllabus"] = bool(uniq_map.get(k, {}).get("syllabus_snippet"))
        if c["extracted_skills"]:
            mapped_cnt += 1

    # Build Dual-Layer Dynamic Knowledge Graph artifact
    kg = {
        "schema_version": "evoalign-kg-2.0",
        "lifecycle_states": ["Emerging", "Incubating", "Core", "Declining", "TeachOut", "Sunset"],
        "skills": CANONICAL_SKILLS,
        "skill_ids": skill_ids,
        "skill_prereq_edges": [{"from": u, "to": v} for u, v in SKILL_PREREQ_EDGES],
        "semantic_ground_distance": {
            "source": emb_src,
            "skill_ids": skill_ids,
            "matrix": dist_matrix,
        },
        "extraction_reliability": reliability,
        "programs": programs,
        "course_skill_catalog": extracted_map,
        "stats": {
            "universities": len({p["university"] for p in programs}),
            "programs": len(programs),
            "total_courses": len(courses),
            "mapped_tech_courses": mapped_cnt,
            "unique_tech_courses": len(unique_list),
            "syllabus_enriched_unique": syl_matched,
            "canonical_skills": len(CANONICAL_SKILLS),
            "prereq_edges": len(SKILL_PREREQ_EDGES),
        },
    }

    (PROC_DIR / "dynamic_kg.json").write_text(json.dumps(kg, ensure_ascii=False, indent=2), encoding="utf-8")
    (PROC_DIR / "university_curricula_enriched.json").write_text(json.dumps(courses, ensure_ascii=False, indent=2), encoding="utf-8")
    print("KG_BUILT_OK:", json.dumps(kg["stats"], ensure_ascii=False))


if __name__ == "__main__":
    main()
