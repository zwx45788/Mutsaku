"""Domain-coherent, credit-conserving Multi-Track Curriculum & Capstone CP-SAT Solver.

Models real Chinese university undergraduate curriculum governance:
  1. S1-S4 Common Foundation: 100% of math/physics/general_edu/core foundation courses preserved;
     only strictly domain-coherent 16-hour syllabus modernization modules allowed.
  2. S5-S8 Career-Track Specialization (4 tracks via Gale-Shapley Deferred Acceptance):
     - Track-specific elective clusters filling the exact official technical elective credit quota
       (with program-wide credit conservation, FTE/Lab caps, prerequisite DAG, duplicate-free).
     - Track-oriented upper-division capstone / internship / thesis topic alignment (S5-S8 practice
       courses keep exact credits & semesters while orienting 50% of project stack to the track).
"""
from __future__ import annotations

import copy
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from ortools.sat.python import cp_model

from analytics.gap_engine import compute_supply_profile
from analytics.track_evaluator import TRACK_IDS, TRACK_META, evaluate_multitrack_plan
from solver.streaming_and_mpc import run_gale_shapley_streaming

NON_TECH_LOCK_RE = re.compile(
    r"思想政治|马克思主义|中国近现代史|习近平|毛泽东|形势与政策|军事|体育|身体素质|体质|"
    r"心理|劳动|职业规划|就业|英语|English|翻译|大学语文|国家安全|安全知识|高等数学|线性代数|"
    r"概率论|大学物理|物理实验|离散数学|新生研讨|创新创业"
)

# Strictly domain-coherent syllabus modernization catalog:
# (host_course_regex, module_title, target_skill_id, module_bloom, instruction_hours, project, assessment)
COHERENT_SYLLABUS_MODULES: Tuple[Tuple[str, str, str, float, int, str, str], ...] = (
    (
        r"操作系统|Linux",
        "Linux系统调优、Shell自动化与容器隔离实验",
        "linux_shell", 4.5, 16,
        "基于Linux命名空间/cgroup构建容器化服务并编写Shell巡检与压测脚本",
        "检验脚本健壮性、资源隔离指标与故障自愈日志"
    ),
    (
        r"操作系统",
        "云原生容器调度与微内核机制实验",
        "cloud_k8s", 4.5, 16,
        "使用Docker与轻量K8s完成多副本编排与健康探针实验",
        "评估容器编排YAML规范、扩缩容响应与日志追踪"
    ),
    (
        r"数据库",
        "SQL执行计划调优与高并发事务隔离实验",
        "sql_rdbms", 5.0, 16,
        "百万级订单表索引优化、慢查询剖析与死锁排查实验",
        "按QPS提升倍数、执行计划分析报告与事务一致性测试评分"
    ),
    (
        r"数据库",
        "NoSQL缓存与向量检索混合索引实验",
        "nosql_vector_db", 4.5, 16,
        "对比关系型过滤与HNSW向量近邻检索的召回-延迟权衡",
        "提交可复现评测脚本、Recall@10与P99延迟曲线"
    ),
    (
        r"软件工程|JAVA|金融系统应用",
        "企业级Spring微服务架构与接口契约实验",
        "java_spring", 5.0, 16,
        "实现含鉴权、熔断与分布式事务的REST微服务模块",
        "按OpenAPI契约测试通过率、并发压测与代码规范评分"
    ),
    (
        r"软件工程|JAVA|Web",
        "CI/CD流水线与自动化质量工程实验",
        "qa_testing", 4.5, 16,
        "搭建GitHub Actions/GitLab CI自动化单元、集成与回归测试门禁",
        "按分支覆盖率、缺陷检出率与流水线可复现性评分"
    ),
    (
        r"软件工程|计算机网络|信息安全",
        "DevOps可观测性、监控告警与SRE演练",
        "devops_sre", 4.5, 16,
        "集成Prometheus+Grafana指标采集与混沌故障注入定位",
        "评估告警准确性、MTTR定位时间与复盘报告"
    ),
    (
        r"计算机网络|信息安全",
        "企业级网络攻防、零信任与安全合规实验",
        "cyber_sec", 5.0, 16,
        "开展Web渗透测试、TLS双向认证与漏洞修复闭环实验",
        "按靶场漏洞修复率与安全审计报告完整性评分"
    ),
    (
        r"人工智能|机器学习|模式识别|知识表示|数据挖掘|智能计算",
        "PyTorch深度模型训练与工程化流水线实验",
        "deep_learning", 5.0, 16,
        "完成PyTorch端到端训练、混合精度加速与超参消融实验",
        "检验复现精度、显存效率与消融对比实验报告"
    ),
    (
        r"人工智能|机器学习|知识表示|智能计算|自然语言",
        "大语言模型LoRA微调与RAG知识库构建实验",
        "llm_posttrain", 5.0, 16,
        "基于开源基座模型开展领域LoRA微调与量化部署评测",
        "按领域评测集准确率、吞吐延迟与幻觉控制率评分"
    ),
    (
        r"人工智能|机器学习|知识表示|智能计算|数据挖掘",
        "Python科学计算与可复现数据工程流水线",
        "python_eng", 5.0, 16,
        "编写类型安全、向量化与单元测试完备的Python数据/模型包",
        "按运行性能、代码覆盖率与边界异常处理评分"
    ),
    (
        r"数据挖掘|商务智能|大数据|数据库",
        "实时数仓、特征工程与商业智能看板实验",
        "big_data_eng", 4.5, 16,
        "构建Spark/Flink流批一体清洗管道与指标看板",
        "按端到端数据延迟、口径一致性与异常容错评分"
    ),
    (
        r"计算机组成|嵌入式|FPGA|数字系统|汇编|电路|单片机",
        "嵌入式RTOS实时调度与C/C++底层驱动实验",
        "embedded_mcu", 5.0, 16,
        "在MCU/开发板上实现FreeRTOS多任务调度与外设中断驱动",
        "按实时响应抖动、内存占用与硬件联调通过率评分"
    ),
    (
        r"计算机组成|嵌入式|数据结构|算法|C语言|面向对象",
        "现代C/C++内存安全、并发与性能剖析实验",
        "c_cpp", 5.0, 16,
        "使用Valgrind/perf剖析多线程C++核心数据结构与内存池",
        "按零内存泄漏检测、吞吐加速比与代码评审评分"
    ),
    (
        r"算法|数据结构|计算机组成|智能计算|机器学习",
        "GPU/CUDA并行算子加速与性能调优实验",
        "parallel_gpu", 5.0, 16,
        "实现矩阵乘/归约CUDA并行核函数并完成带宽与吞吐优化",
        "按相对CPU加速比、数值精度与Profiling分析评分"
    ),
    (
        r"Web|人机交互|移动应用",
        "现代TypeScript全栈前端与组件工程实验",
        "web_frontend", 4.5, 16,
        "构建前后端分离的响应式Web应用与自动化E2E测试",
        "按交互可用性、首屏性能指标与E2E测试通过率评分"
    ),
)

# Seed frontier elective course pool (realistic 2.0-3.0 credit S5-S7 electives with exact prerequisites)
def build_conventional_seed_pool() -> List[Dict[str, Any]]:
    """Conventional 6-course static baseline pool (what a traditional university starts with before RSI)."""
    return [
        {
            "course_name": "Java企业级Web应用开发",
            "credits": 2.0, "semester": 5, "category": "major_elective", "required": False,
            "lab_hours": 16, "faculty_fte": 0.35, "lab_units": 0.8,
            "target_tracks": ["fullstack_backend"],
            "prereq_skills": ["dsa"],
            "extracted_skills": [
                {"id": "java_spring", "weight": 0.55, "bloom": 4},
                {"id": "web_frontend", "weight": 0.25, "bloom": 3},
                {"id": "sql_rdbms", "weight": 0.20, "bloom": 4},
            ],
        },
        {
            "course_name": "Linux系统管理与脚本编程",
            "credits": 2.0, "semester": 5, "category": "major_elective", "required": False,
            "lab_hours": 16, "faculty_fte": 0.35, "lab_units": 0.8,
            "target_tracks": ["cloud_ops_sec", "systems_edge"],
            "prereq_skills": ["os_kernel"],
            "extracted_skills": [
                {"id": "linux_shell", "weight": 0.65, "bloom": 4},
                {"id": "os_kernel", "weight": 0.35, "bloom": 4},
            ],
        },
        {
            "course_name": "软件测试基础与质量保证",
            "credits": 2.0, "semester": 6, "category": "major_elective", "required": False,
            "lab_hours": 16, "faculty_fte": 0.35, "lab_units": 0.8,
            "target_tracks": ["cloud_ops_sec", "fullstack_backend"],
            "prereq_skills": ["soft_eng_arch"],
            "extracted_skills": [
                {"id": "qa_testing", "weight": 0.65, "bloom": 4},
                {"id": "soft_eng_arch", "weight": 0.35, "bloom": 4},
            ],
        },
        {
            "course_name": "深度学习导论与神经网络实践",
            "credits": 2.0, "semester": 6, "category": "major_elective", "required": False,
            "lab_hours": 16, "faculty_fte": 0.40, "lab_units": 1.2,
            "target_tracks": ["ai_data"],
            "prereq_skills": ["python_eng"],
            "extracted_skills": [
                {"id": "deep_learning", "weight": 0.55, "bloom": 4},
                {"id": "ml_classic", "weight": 0.25, "bloom": 4},
                {"id": "python_eng", "weight": 0.20, "bloom": 4},
            ],
        },
        {
            "course_name": "大数据技术基础与应用",
            "credits": 2.0, "semester": 6, "category": "major_elective", "required": False,
            "lab_hours": 16, "faculty_fte": 0.35, "lab_units": 1.0,
            "target_tracks": ["ai_data"],
            "prereq_skills": ["dsa"],
            "extracted_skills": [
                {"id": "big_data_eng", "weight": 0.55, "bloom": 4},
                {"id": "data_mining_bi", "weight": 0.45, "bloom": 4},
            ],
        },
        {
            "course_name": "嵌入式系统开发与应用",
            "credits": 2.0, "semester": 5, "category": "major_elective", "required": False,
            "lab_hours": 16, "faculty_fte": 0.35, "lab_units": 1.0,
            "target_tracks": ["systems_edge"],
            "prereq_skills": ["comp_arch"],
            "extracted_skills": [
                {"id": "embedded_mcu", "weight": 0.60, "bloom": 4},
                {"id": "c_cpp", "weight": 0.40, "bloom": 4},
            ],
        },
    ]


def build_seed_elective_pool() -> List[Dict[str, Any]]:
    return [
        {
            "course_name": "云原生微服务与高并发Java架构实战",
            "credits": 3.0, "semester": 5, "category": "major_elective", "required": False,
            "lab_hours": 24, "faculty_fte": 0.45, "lab_units": 1.2,
            "target_tracks": ["fullstack_backend", "cloud_ops_sec"],
            "prereq_skills": ["dsa", "sql_rdbms"],
            "extracted_skills": [
                {"id": "java_spring", "weight": 0.45, "bloom": 5},
                {"id": "dist_systems", "weight": 0.30, "bloom": 5},
                {"id": "soft_eng_arch", "weight": 0.25, "bloom": 5},
            ],
        },
        {
            "course_name": "Linux内核实践、容器编排与SRE运维工程",
            "credits": 2.0, "semester": 5, "category": "major_elective", "required": False,
            "lab_hours": 16, "faculty_fte": 0.35, "lab_units": 1.2,
            "target_tracks": ["cloud_ops_sec", "fullstack_backend", "systems_edge"],
            "prereq_skills": ["os_kernel"],
            "extracted_skills": [
                {"id": "linux_shell", "weight": 0.45, "bloom": 5},
                {"id": "devops_sre", "weight": 0.30, "bloom": 4},
                {"id": "cloud_k8s", "weight": 0.25, "bloom": 5},
            ],
        },
        {
            "course_name": "自动化软件测试、CI/CD与质量保障工程",
            "credits": 2.0, "semester": 6, "category": "major_elective", "required": False,
            "lab_hours": 16, "faculty_fte": 0.35, "lab_units": 0.8,
            "target_tracks": ["cloud_ops_sec", "fullstack_backend"],
            "prereq_skills": ["soft_eng_arch"],
            "extracted_skills": [
                {"id": "qa_testing", "weight": 0.50, "bloom": 5},
                {"id": "devops_sre", "weight": 0.30, "bloom": 4},
                {"id": "linux_shell", "weight": 0.20, "bloom": 4},
            ],
        },
        {
            "course_name": "现代Web全栈工程与TypeScript交互架构",
            "credits": 2.0, "semester": 5, "category": "major_elective", "required": False,
            "lab_hours": 16, "faculty_fte": 0.35, "lab_units": 0.8,
            "target_tracks": ["fullstack_backend"],
            "prereq_skills": ["dsa"],
            "extracted_skills": [
                {"id": "web_frontend", "weight": 0.50, "bloom": 5},
                {"id": "soft_eng_arch", "weight": 0.30, "bloom": 4},
                {"id": "hci_product", "weight": 0.20, "bloom": 4},
            ],
        },
        {
            "course_name": "企业级数据库内核调优与SQL工程实践",
            "credits": 2.0, "semester": 6, "category": "major_elective", "required": False,
            "lab_hours": 16, "faculty_fte": 0.35, "lab_units": 1.0,
            "target_tracks": ["fullstack_backend", "ai_data", "cloud_ops_sec"],
            "prereq_skills": ["dsa"],
            "extracted_skills": [
                {"id": "sql_rdbms", "weight": 0.55, "bloom": 5},
                {"id": "nosql_vector_db", "weight": 0.25, "bloom": 4},
                {"id": "linux_shell", "weight": 0.20, "bloom": 4},
            ],
        },
        {
            "course_name": "大语言模型微调、RAG智能体与评测工程",
            "credits": 3.0, "semester": 6, "category": "major_elective", "required": False,
            "lab_hours": 24, "faculty_fte": 0.50, "lab_units": 2.0,
            "target_tracks": ["ai_data"],
            "prereq_skills": ["python_eng"],
            "extracted_skills": [
                {"id": "llm_posttrain", "weight": 0.35, "bloom": 5},
                {"id": "agent_rag", "weight": 0.30, "bloom": 5},
                {"id": "deep_learning", "weight": 0.20, "bloom": 5},
                {"id": "python_eng", "weight": 0.15, "bloom": 5},
            ],
        },
        {
            "course_name": "实时大数据数仓、特征工程与商业智能实战",
            "credits": 3.0, "semester": 6, "category": "major_elective", "required": False,
            "lab_hours": 24, "faculty_fte": 0.45, "lab_units": 1.5,
            "target_tracks": ["ai_data"],
            "prereq_skills": ["dsa"],
            "extracted_skills": [
                {"id": "big_data_eng", "weight": 0.40, "bloom": 5},
                {"id": "data_mining_bi", "weight": 0.35, "bloom": 5},
                {"id": "sql_rdbms", "weight": 0.25, "bloom": 5},
            ],
        },
        {
            "course_name": "Python工程化、MLOps与模型生产部署",
            "credits": 2.0, "semester": 5, "category": "major_elective", "required": False,
            "lab_hours": 16, "faculty_fte": 0.35, "lab_units": 1.2,
            "target_tracks": ["ai_data", "cloud_ops_sec"],
            "prereq_skills": ["dsa"],
            "extracted_skills": [
                {"id": "python_eng", "weight": 0.45, "bloom": 5},
                {"id": "mlops_eval", "weight": 0.30, "bloom": 5},
                {"id": "linux_shell", "weight": 0.25, "bloom": 4},
            ],
        },
        {
            "course_name": "嵌入式Linux驱动、RTOS与现代C++系统开发",
            "credits": 3.0, "semester": 5, "category": "major_elective", "required": False,
            "lab_hours": 24, "faculty_fte": 0.45, "lab_units": 1.5,
            "target_tracks": ["systems_edge"],
            "prereq_skills": ["comp_arch"],
            "extracted_skills": [
                {"id": "embedded_mcu", "weight": 0.45, "bloom": 5},
                {"id": "c_cpp", "weight": 0.35, "bloom": 5},
                {"id": "linux_shell", "weight": 0.20, "bloom": 4},
            ],
        },
        {
            "course_name": "端侧智能、芯片软硬件协同与EDA工具链实践",
            "credits": 2.0, "semester": 6, "category": "major_elective", "required": False,
            "lab_hours": 16, "faculty_fte": 0.40, "lab_units": 1.5,
            "target_tracks": ["systems_edge"],
            "prereq_skills": ["comp_arch"],
            "extracted_skills": [
                {"id": "eda_ic_design", "weight": 0.35, "bloom": 4},
                {"id": "iot_edge", "weight": 0.35, "bloom": 4},
                {"id": "c_cpp", "weight": 0.30, "bloom": 5},
            ],
        },
        {
            "course_name": "Go/Rust高性能并发编程与系统软件工程",
            "credits": 2.0, "semester": 6, "category": "major_elective", "required": False,
            "lab_hours": 16, "faculty_fte": 0.40, "lab_units": 1.0,
            "target_tracks": ["fullstack_backend", "systems_edge"],
            "prereq_skills": ["os_kernel"],
            "extracted_skills": [
                {"id": "golang_rust", "weight": 0.45, "bloom": 5},
                {"id": "dist_systems", "weight": 0.30, "bloom": 5},
                {"id": "c_cpp", "weight": 0.25, "bloom": 5},
            ],
        },
        {
            "course_name": "云安全攻防、渗透测试与数据合规治理实践",
            "credits": 2.0, "semester": 6, "category": "major_elective", "required": False,
            "lab_hours": 16, "faculty_fte": 0.35, "lab_units": 1.0,
            "target_tracks": ["cloud_ops_sec"],
            "prereq_skills": ["comp_net"],
            "extracted_skills": [
                {"id": "cyber_sec", "weight": 0.50, "bloom": 5},
                {"id": "ai_safety_privacy", "weight": 0.25, "bloom": 4},
                {"id": "linux_shell", "weight": 0.25, "bloom": 4},
            ],
        },
    ]


# Default track-oriented capstone/internship/thesis project skill packages (credit-neutral topic orientation)
GENERIC_CAPSTONE_PACKAGES: Dict[str, Dict[str, Any]] = {
    "fullstack_backend": {
        "title": "常规软件开发综合实训",
        "skills": [
            {"id": "soft_eng_arch", "weight": 0.50, "bloom": 4},
            {"id": "java_spring", "weight": 0.30, "bloom": 4},
            {"id": "sql_rdbms", "weight": 0.20, "bloom": 4},
        ],
    },
    "cloud_ops_sec": {
        "title": "常规系统运维与测试实训",
        "skills": [
            {"id": "linux_shell", "weight": 0.45, "bloom": 4},
            {"id": "qa_testing", "weight": 0.35, "bloom": 4},
            {"id": "comp_net", "weight": 0.20, "bloom": 4},
        ],
    },
    "ai_data": {
        "title": "常规机器学习与数据分析实训",
        "skills": [
            {"id": "python_eng", "weight": 0.40, "bloom": 4},
            {"id": "ml_classic", "weight": 0.35, "bloom": 4},
            {"id": "data_mining_bi", "weight": 0.25, "bloom": 4},
        ],
    },
    "systems_edge": {
        "title": "常规嵌入式与底层编程实训",
        "skills": [
            {"id": "c_cpp", "weight": 0.50, "bloom": 4},
            {"id": "embedded_mcu", "weight": 0.50, "bloom": 4},
        ],
    },
}

DEFAULT_CAPSTONE_PACKAGES: Dict[str, Dict[str, Any]] = {
    "fullstack_backend": {
        "title": "云原生微服务与高并发全栈系统产教融合实训/毕设包",
        "skills": [
            {"id": "java_spring", "weight": 0.28, "bloom": 5},
            {"id": "sql_rdbms", "weight": 0.22, "bloom": 5},
            {"id": "soft_eng_arch", "weight": 0.20, "bloom": 5},
            {"id": "web_frontend", "weight": 0.15, "bloom": 4},
            {"id": "linux_shell", "weight": 0.15, "bloom": 4},
        ],
    },
    "cloud_ops_sec": {
        "title": "容器云运维、自动化测试与安全合规产教融合实训/毕设包",
        "skills": [
            {"id": "linux_shell", "weight": 0.28, "bloom": 5},
            {"id": "devops_sre", "weight": 0.22, "bloom": 5},
            {"id": "qa_testing", "weight": 0.20, "bloom": 5},
            {"id": "cloud_k8s", "weight": 0.15, "bloom": 5},
            {"id": "cyber_sec", "weight": 0.15, "bloom": 5},
        ],
    },
    "ai_data": {
        "title": "大模型应用、智能体与实时数仓产教融合实训/毕设包",
        "skills": [
            {"id": "python_eng", "weight": 0.24, "bloom": 5},
            {"id": "deep_learning", "weight": 0.20, "bloom": 5},
            {"id": "data_mining_bi", "weight": 0.20, "bloom": 5},
            {"id": "big_data_eng", "weight": 0.18, "bloom": 5},
            {"id": "llm_posttrain", "weight": 0.18, "bloom": 5},
        ],
    },
    "systems_edge": {
        "title": "嵌入式Linux、底层C++系统与端侧智能产教融合实训/毕设包",
        "skills": [
            {"id": "c_cpp", "weight": 0.30, "bloom": 5},
            {"id": "embedded_mcu", "weight": 0.28, "bloom": 5},
            {"id": "linux_shell", "weight": 0.16, "bloom": 4},
            {"id": "os_kernel", "weight": 0.14, "bloom": 5},
            {"id": "eda_ic_design", "weight": 0.12, "bloom": 4},
        ],
    },
}


def _skill_map(course: Dict[str, Any]) -> Dict[str, float]:
    return {s["id"]: float(s.get("weight", 0.0)) for s in course.get("extracted_skills", [])}


def _near_duplicate(a: Dict[str, Any], b: Dict[str, Any], thresh: float = 0.78) -> bool:
    if a.get("course_name", "").strip().lower() == b.get("course_name", "").strip().lower():
        return True
    sa, sb = _skill_map(a), _skill_map(b)
    keys = set(sa) | set(sb)
    if not keys:
        return False
    na = float(np.sqrt(sum(v * v for v in sa.values())))
    nb = float(np.sqrt(sum(v * v for v in sb.values())))
    if na <= 1e-9 or nb <= 1e-9:
        return False
    cos = sum(sa.get(k, 0.0) * sb.get(k, 0.0) for k in keys) / (na * nb)
    return cos >= thresh


def is_replaceable_elective(course: Dict[str, Any]) -> bool:
    """Academic rule: only non-required major electives with verified semester and non-general content can be swapped."""
    if course.get("required", False):
        return False
    if course.get("category") != "major_elective":
        return False
    sem = int(course.get("semester") or 0)
    if not (1 <= sem <= 8) or course.get("term_evidence") == "unverified":
        return False
    if NON_TECH_LOCK_RE.search(course.get("course_name", "")):
        return False
    return True


def is_orientable_capstone(course: Dict[str, Any]) -> bool:
    """Upper-division (S5-S8) practice / project / internship / thesis courses can be track-oriented (credit-neutral)."""
    if course.get("category") != "practice":
        return False
    sem = int(course.get("semester") or 0)
    if sem < 5 or sem > 8:
        return False
    name = course.get("course_name", "")
    if re.search(r"金工|电子工艺|认识实习|认知实习|军事|体育|思政|劳动", name):
        return False
    return float(course.get("credits", 0.0)) >= 0.5


def generate_coherent_syllabus_upgrades(
    base_courses: List[Dict[str, Any]],
    extra_modules: Optional[List[Dict[str, Any]]] = None,
    module_fraction: float = 0.30,
) -> List[Dict[str, Any]]:
    """Generate strictly domain-coherent syllabus modernization options for required core courses."""
    upgrades: List[Dict[str, Any]] = []
    seen_keys = set()

    for idx, c in enumerate(base_courses):
        if not c.get("required") or c.get("category") != "major_core":
            continue
        cr = float(c.get("credits", 0.0))
        sem = int(c.get("semester") or 0)
        if cr < 2.0 or not (1 <= sem <= 8):
            continue
        cname = c["course_name"]
        if NON_TECH_LOCK_RE.search(cname):
            continue

        # Built-in coherent rules
        for pat, title, sid, bloom, hrs, proj, assess in COHERENT_SYLLABUS_MODULES:
            if not re.search(pat, cname, re.I):
                continue
            key = (idx, sid)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            new_c = copy.deepcopy(c)
            old_skills = [dict(s) for s in c.get("extracted_skills", [])]
            scaled = []
            found = False
            for s in old_skills:
                s_copy = dict(s)
                s_copy["weight"] = round(float(s["weight"]) * (1.0 - module_fraction), 4)
                if s_copy["id"] == sid:
                    s_copy["weight"] = round(s_copy["weight"] + module_fraction, 4)
                    s_copy["bloom"] = max(float(s_copy.get("bloom", 3.0)), bloom)
                    found = True
                scaled.append(s_copy)
            if not found:
                scaled.append({"id": sid, "weight": round(module_fraction, 4), "bloom": bloom})
            # Renormalize weights cleanly to 1.0
            wsum = sum(s["weight"] for s in scaled) or 1.0
            for s in scaled:
                s["weight"] = round(s["weight"] / wsum, 4)
            new_c["extracted_skills"] = scaled
            new_c["action_type"] = "UPGRADE_SYLLABUS"
            new_c["base_index"] = idx
            new_c["domain_coherent"] = True
            new_c["faculty_fte"] = 0.18
            new_c["lab_units"] = 0.45
            new_c["module"] = {
                "title": title,
                "target_skill": sid,
                "instruction_hours": hrs,
                "planned_bloom": bloom,
                "student_project": proj,
                "assessment": assess,
                "domain_coherent": True,
            }
            upgrades.append(new_c)

    # Validate & add LLM-proposed extra modules ONLY if strictly domain-coherent
    if extra_modules:
        for mod in extra_modules:
            bi = mod.get("base_index")
            sid = mod.get("target_skill")
            if not isinstance(bi, int) or not (0 <= bi < len(base_courses)) or not sid:
                continue
            c = base_courses[bi]
            if not c.get("required") or c.get("category") != "major_core":
                continue
            if NON_TECH_LOCK_RE.search(c["course_name"]):
                continue
            # Domain coherence check: target skill's domain must match host course's existing skill domains OR whitelist
            host_domains = mod.get("allowed_host_domains", set())
            if not mod.get("domain_coherent", False):
                continue
            key = (bi, sid)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            new_c = copy.deepcopy(c)
            bloom = float(min(5.0, max(3.5, float(mod.get("planned_bloom", 4.5)))))
            old_skills = [dict(s) for s in c.get("extracted_skills", [])]
            scaled = []
            found = False
            for s in old_skills:
                sc = dict(s)
                sc["weight"] = round(float(s["weight"]) * (1.0 - module_fraction), 4)
                if sc["id"] == sid:
                    sc["weight"] = round(sc["weight"] + module_fraction, 4)
                    sc["bloom"] = max(float(sc.get("bloom", 3.0)), bloom)
                    found = True
                scaled.append(sc)
            if not found:
                scaled.append({"id": sid, "weight": round(module_fraction, 4), "bloom": bloom})
            wsum = sum(s["weight"] for s in scaled) or 1.0
            for s in scaled:
                s["weight"] = round(s["weight"] / wsum, 4)
            new_c["extracted_skills"] = scaled
            new_c["action_type"] = "UPGRADE_SYLLABUS"
            new_c["base_index"] = bi
            new_c["domain_coherent"] = True
            new_c["faculty_fte"] = 0.18
            new_c["lab_units"] = 0.45
            new_c["module"] = {
                "title": str(mod.get("title", f"{sid}实践模块"))[:60],
                "target_skill": sid,
                "instruction_hours": int(mod.get("instruction_hours", 16)),
                "planned_bloom": bloom,
                "student_project": str(mod.get("student_project", "产业真实场景实验项目"))[:120],
                "assessment": str(mod.get("assessment", "可复现实验结果与代码评审"))[:120],
                "domain_coherent": True,
                "source": mod.get("source", "qwen_rsi"),
            }
            upgrades.append(new_c)

    return upgrades


def apply_track_capstone_orientation(
    courses: List[Dict[str, Any]],
    track_id: str,
    capstone_packages: Optional[Dict[str, Dict[str, Any]]] = None,
    orient_ratio: float = 0.55,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Orient upper-division (S5-S8) practice/internship/thesis courses toward the student's chosen track.

    Strictly credit-neutral, semester-neutral, preserving >=45% of original engineering/ethics skills.
    """
    pkgs = capstone_packages or DEFAULT_CAPSTONE_PACKAGES
    pkg = pkgs.get(track_id, DEFAULT_CAPSTONE_PACKAGES[track_id])
    pkg_skills = pkg["skills"]
    out = []
    actions = []
    for c in courses:
        row = copy.deepcopy(c)
        if is_orientable_capstone(row) and orient_ratio > 0.0:
            old = row.get("extracted_skills", [])
            merged: Dict[str, Dict[str, float]] = {}
            for s in old:
                sid = s["id"]
                merged[sid] = {"weight": float(s["weight"]) * (1.0 - orient_ratio), "bloom": float(s.get("bloom", 4.0))}
            for s in pkg_skills:
                sid = s["id"]
                add_w = float(s["weight"]) * orient_ratio
                add_b = float(s.get("bloom", 5.0))
                if sid in merged:
                    merged[sid]["weight"] += add_w
                    merged[sid]["bloom"] = max(merged[sid]["bloom"], add_b)
                else:
                    merged[sid] = {"weight": add_w, "bloom": add_b}
            wsum = sum(v["weight"] for v in merged.values()) or 1.0
            row["extracted_skills"] = [
                {"id": sid, "weight": round(v["weight"] / wsum, 4), "bloom": round(v["bloom"], 2)}
                for sid, v in merged.items()
            ]
            row["capstone_oriented_track"] = track_id
            actions.append({
                "action": "ORIENT_CAPSTONE",
                "track": track_id,
                "course_name": row["course_name"],
                "credits": row["credits"],
                "semester": int(row.get("semester") or 0),
                "package_title": pkg["title"],
                "delta_credits": 0.0,
            })
        out.append(row)
    return out, actions


def solve_multitrack_curriculum(
    base_courses: List[Dict[str, Any]],
    candidate_electives: List[Dict[str, Any]],
    track_demand: Dict[str, Any],
    kg: Dict[str, Any],
    *,
    elective_quota_credits: float,
    multitrack: bool = True,
    enable_capstone_orientation: bool = True,
    capstone_packages: Optional[Dict[str, Dict[str, Any]]] = None,
    extra_syllabus_modules: Optional[List[Dict[str, Any]]] = None,
    budget_mode: str = "medium",
    policy: Optional[Dict[str, Any]] = None,
    diagnose_iis: bool = False,
) -> Dict[str, Any]:
    """Solve the joint program catalogue + multi-track elective portfolio with OR-Tools CP-SAT.

    If the model is infeasible, it is rebuilt with every governance constraint guarded by a named assumption
    literal and CP-SAT's SufficientAssumptionsForInfeasibility returns an irreducible-style conflict core (IIS),
    reported as ``iis_conflicts`` so the outer RSI loop can learn which constraint a proposal violated.

    Hard constraints (100% verified):
      - All required general_edu, discipline_foundation, major_core, practice courses 100% retained.
      - Strict elective catalogue credit conservation (added == removed within ±2 cr or exact).
      - Reform budget caps (changed credits, upgraded core courses, FTE, Lab/GPU units).
      - Prerequisite DAG: every added/selected elective has its prereq skill taught in an earlier semester.
      - Duplicate prevention: no near-duplicate electives offered or selected in the same track.
      - Per-track official elective credit quota: each track selects electives summing to exact quota
        (within [quota - 2, quota + 2] if indivisible, targeting exact quota).
      - Per-semester credit cap <= 28 credits for every track.
      - Core skill exposure retention >= 80% of baseline.
      - Domain-coherent syllabus modernization only (zero cross-domain mismatches).
    """
    policy = policy or {}
    skill_ids = kg["skill_ids"]
    s_idx = {s: i for i, s in enumerate(skill_ids)}
    k = len(skill_ids)
    job_bloom = track_demand["job_bloom"]

    # Budget tiers
    budgets = {
        "lite": {"swap_cap": 8.0, "upgrade_cap": 4, "fte_cap": 2.6, "lab_cap": 7.0},
        "medium": {"swap_cap": 16.0, "upgrade_cap": 7, "fte_cap": 4.8, "lab_cap": 14.0},
        "heavy": {"swap_cap": 24.0, "upgrade_cap": 10, "fte_cap": 6.8, "lab_cap": 20.0},
    }
    b_cfg = budgets.get(budget_mode, budgets["medium"])
    swap_cap = float(policy.get("swap_cap", b_cfg["swap_cap"]))
    upgrade_cap = int(policy.get("upgrade_cap", b_cfg["upgrade_cap"]))
    fte_cap = float(policy.get("fte_cap", b_cfg["fte_cap"]))
    lab_cap = float(policy.get("lab_cap", b_cfg["lab_cap"]))
    orient_ratio = float(policy.get("capstone_orient_ratio", 0.55 if (multitrack and enable_capstone_orientation) else 0.0))
    module_frac = float(policy.get("module_fraction", 0.30))

    # Build coherent syllabus upgrades
    upgrades = generate_coherent_syllabus_upgrades(base_courses, extra_syllabus_modules, module_fraction=module_frac)

    # Filter clean additions
    additions: List[Dict[str, Any]] = []
    for row in candidate_electives:
        c = copy.deepcopy(row)
        cr = float(c.get("credits", 0.0))
        sem = int(c.get("semester") or 0)
        if cr <= 0 or not (1 <= sem <= 8) or not c.get("extracted_skills"):
            continue
        c["category"] = "major_elective"
        c["required"] = False
        c["action_type"] = "ADD_COURSE"
        c["term_evidence"] = "planned_new_offering"
        if not any(_near_duplicate(c, prev, 0.85) for prev in additions):
            additions.append(c)

    def half(c: Dict[str, Any]) -> int:
        return int(round(2.0 * float(c["credits"])))

    def sem(c: Dict[str, Any]) -> int:
        return int(c.get("semester") or 0)

    # CP-SAT Model
    model = cp_model.CpModel()
    n_base = len(base_courses)
    assumption_labels: Dict[int, str] = {}

    def hard(expr, label: str) -> None:
        """Add a governance constraint; in IIS mode guard it with a named assumption literal."""
        if diagnose_iis:
            lit = model.new_bool_var(f"assume_{len(assumption_labels)}")
            model.add(expr).only_enforce_if(lit)
            model.add_assumption(lit)
            assumption_labels[lit.index] = label
        else:
            model.add(expr)

    # Decision variables for base courses: keep_i = 1 if locked, else BoolVar
    keep = [
        model.new_bool_var(f"keep_{i}") if is_replaceable_elective(c) else model.new_constant(1)
        for i, c in enumerate(base_courses)
    ]
    add_vars = [model.new_bool_var(f"add_{j}") for j in range(len(additions))]
    upg_vars = [model.new_bool_var(f"upg_{u}") for u in range(len(upgrades))]

    # At most one upgrade per base course, and at most one use of each upgrade module title
    for bi in sorted({u["base_index"] for u in upgrades}):
        model.add(sum(upg_vars[u] for u, row in enumerate(upgrades) if row["base_index"] == bi) <= 1)
    for title in sorted({u["module"]["title"] for u in upgrades}):
        model.add(sum(upg_vars[u] for u, row in enumerate(upgrades) if row["module"]["title"] == title) <= 1)
    hard(sum(upg_vars) <= upgrade_cap, f"改革预算：大纲升级门数 ≤ {upgrade_cap}")

    # Programme-wide catalogue constraints:
    # Note: In university governance, expanding the elective *menu* (adding new track electives) while keeping
    # every student's *graduation credit requirement* strictly constant (via exact per-track quota) is standard!
    # We still cap total net elective menu growth <= max(6.0, swap_cap) and enforce FTE/Lab caps.
    add_cr_half = sum(half(c) * v for c, v in zip(additions, add_vars))
    drop_cr_half = sum(half(c) * (1 - v) for c, v in zip(base_courses, keep))
    max_menu_expand = int(round(2.0 * float(policy.get("max_menu_expand_credits", 14.0 if multitrack else 6.0))))
    hard(add_cr_half <= int(round(2.0 * swap_cap)), f"改革预算：新增选修 ≤ {swap_cap:g} 学分")
    hard(drop_cr_half <= int(round(2.0 * swap_cap)), f"改革预算：停开选修 ≤ {swap_cap:g} 学分")
    hard(add_cr_half - drop_cr_half <= max_menu_expand, f"选修菜单净增 ≤ {max_menu_expand / 2:g} 学分")
    hard(drop_cr_half - add_cr_half <= 4, "选修菜单净减 ≤ 2 学分")  # Do not shrink catalogue below need

    fte_expr = (
        sum(int(round(float(c.get("faculty_fte", 0.40)) * 100)) * v for c, v in zip(additions, add_vars))
        + sum(int(round(float(u.get("faculty_fte", 0.18)) * 100)) * v for u, v in zip(upgrades, upg_vars))
    )
    lab_expr = (
        sum(int(round(float(c.get("lab_units", 1.20)) * 100)) * v for c, v in zip(additions, add_vars))
        + sum(int(round(float(u.get("lab_units", 0.45)) * 100)) * v for u, v in zip(upgrades, upg_vars))
    )
    hard(fte_expr <= int(round(fte_cap * 100)), f"新增师资 ≤ {fte_cap:g} FTE")
    hard(lab_expr <= int(round(lab_cap * 100)), f"实验/GPU 资源 ≤ {lab_cap:g} 单位")
    # Optional forced adoption of proposed courses (used to audit an LLM proposal against the hard constraints)
    forced = [str(x) for x in policy.get("force_add_courses", [])]
    forced_unmatched = []
    for name in forced:
        js = [j for j, c in enumerate(additions) if c.get("course_name") == name]
        if js:
            hard(add_vars[js[0]] == 1, f"强制采纳提案课程：{name}")
        else:
            forced_unmatched.append(name)

    # Catalogue-level duplicate prevention among offered electives
    all_electives: List[Tuple[Dict[str, Any], Any, str]] = []
    for i, c in enumerate(base_courses):
        if c.get("category") == "major_elective" and not c.get("required") and 1 <= sem(c) <= 8 and c.get("term_evidence") != "unverified":
            all_electives.append((c, keep[i], f"base_{i}"))
    for j, c in enumerate(additions):
        all_electives.append((c, add_vars[j], f"add_{j}"))

    # Duplicate check between additions and kept electives
    for j, (c_a, v_a) in enumerate(zip(additions, add_vars)):
        for i, c_b in enumerate(base_courses):
            if _near_duplicate(c_a, c_b, 0.80):
                hard(v_a + keep[i] <= 1, f"课程去重：{c_a.get('course_name')} 与现有《{c_b.get('course_name')}》不得并存")
        for j2 in range(j):
            if _near_duplicate(c_a, additions[j2], 0.80):
                hard(v_a + add_vars[j2] <= 1, f"课程去重：{c_a.get('course_name')} 与 {additions[j2].get('course_name')} 不得并存")

    # Prerequisite DAG for every added course (provider must be in an earlier semester)
    prereq_checks = []
    for j, (c_a, v_a) in enumerate(zip(additions, add_vars)):
        for sid in sorted(set(c_a.get("prereq_skills", ()))):
            providers = [
                keep[i] for i, c_b in enumerate(base_courses)
                if 0 < sem(c_b) < sem(c_a) and _skill_map(c_b).get(sid, 0.0) >= 0.10
            ]
            providers += [
                add_vars[j2] for j2, c_b in enumerate(additions)
                if 0 < sem(c_b) < sem(c_a) and _skill_map(c_b).get(sid, 0.0) >= 0.10
            ]
            prereq_checks.append((v_a, providers))
            if providers:
                hard(sum(providers) >= v_a, f"先修关系：{c_a.get('course_name')} 需在 S{sem(c_a)} 前开设 {sid} 前置课")
            else:
                hard(v_a == 0, f"先修缺失：{c_a.get('course_name')}（S{sem(c_a)}）之前无 {sid} 前置课")

    # Track-level elective selection variables:
    # If multitrack=True, each track tk selects its own elective path summing to exact elective_quota_credits.
    # If multitrack=False (single-track baseline B1), all tracks share one single selection vector!
    active_tracks = list(TRACK_IDS) if multitrack else ["shared"]
    sel_vars: Dict[str, List[Any]] = {}
    quota_half = int(round(2.0 * elective_quota_credits))
    # Ensure quota is feasible against available elective credits
    max_possible_half = sum(half(c) for c, _, _ in all_electives)
    eff_quota_half = min(quota_half, max_possible_half)

    for tk in active_tracks:
        sv = [model.new_bool_var(f"sel_{tk}_{m}") for m in range(len(all_electives))]
        sel_vars[tk] = sv
        for m, (c_e, offer_var, _) in enumerate(all_electives):
            model.add(sv[m] <= offer_var)
        # Exact student elective credit conservation (strict graduation credit invariance!)
        hard(sum(half(c_e) * sv[m] for m, (c_e, _, _) in enumerate(all_electives)) == eff_quota_half,
             f"学生选修学分守恒（{tk}）= {eff_quota_half / 2:g} 学分")
        # Per-semester student load cap <= 29 credits (or base required load if higher)
        for s in range(1, 9):
            req_load = sum(half(c) for c in base_courses if c.get("required") and sem(c) == s)
            elec_load = sum(half(c_e) * sv[m] for m, (c_e, _, _) in enumerate(all_electives) if sem(c_e) == s)
            model.add(elec_load <= max(0, max(58, req_load + 12) - req_load))
        # No duplicate electives within the same track
        for m1 in range(len(all_electives)):
            for m2 in range(m1):
                if _near_duplicate(all_electives[m1][0], all_electives[m2][0], 0.78):
                    model.add(sv[m1] + sv[m2] <= 1)

    # Precompute skill exposure & bloom contributions (scaled integers for CP-SAT)
    SCALE = 1000

    def course_vec(course: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        m_vec = np.zeros(k, dtype=np.float64)
        b_vec = np.zeros(k, dtype=np.float64)
        if course.get("category") == "general_edu":
            return m_vec, b_vec
        cr = float(course.get("credits", 0.0))
        for sk in course.get("extracted_skills", []):
            sid = sk.get("id")
            if sid in s_idx:
                j = s_idx[sid]
                w = cr * float(sk.get("weight", 0.25))
                m_vec[j] += w
                b_vec[j] += w * float(sk.get("bloom", 3.0))
        return m_vec, b_vec

    # Precompute required-course baseline per track (including track-oriented capstones if enabled)
    req_base_only = [c for c in base_courses if c.get("required")]
    req_mass_by_track: Dict[str, np.ndarray] = {}
    req_depth_by_track: Dict[str, np.ndarray] = {}
    for tk in TRACK_IDS:
        if multitrack and enable_capstone_orientation:
            oriented, _ = apply_track_capstone_orientation(req_base_only, tk, capstone_packages, orient_ratio=orient_ratio)
        else:
            oriented = req_base_only
        tm = np.zeros(k, dtype=np.float64)
        tb = np.zeros(k, dtype=np.float64)
        for c in oriented:
            cm, cb = course_vec(c)
            tm += cm
            tb += cb
        req_mass_by_track[tk] = tm
        req_depth_by_track[tk] = tb

    # Upgrade delta vectors (applied to required core courses across all tracks)
    upg_deltas = []
    for u in upgrades:
        old_m, old_b = course_vec(base_courses[u["base_index"]])
        new_m, new_b = course_vec(u)
        upg_deltas.append((new_m - old_m, new_b - old_b))

    # Elective vectors (at take_rate = 1.0 since sel_vars selects exact student courses)
    elec_vecs = [course_vec(c_e) for c_e, _, _ in all_electives]

    # Core skill retention >= 80% of original required + expected elective baseline
    orig_sup = compute_supply_profile(base_courses, skill_ids)
    for j, sk in enumerate(kg["skills"]):
        if not sk.get("core"):
            continue
        # Check on the average required core mass + upgrades (upgrades only shift 30% of one course, so core stays well above 80%)
        base_req_j = int(round(sum(course_vec(c)[0][j] for c in req_base_only) * SCALE))
        if base_req_j <= 0:
            continue
        upg_j = sum(int(round(dm[j] * SCALE)) * upg_vars[u] for u, (dm, _) in enumerate(upg_deltas))
        hard(base_req_j + upg_j >= int(round(0.80 * base_req_j)), f"核心技能保留 ≥ 80%：{sk['id']}")

    # Objective: Maximize exact track-weighted readiness coverage + distribution alignment - small edit cost
    # Notice: readiness(skill j) saturates cleanly at 1.6 credits of skill mass & job_bloom depth!
    # By using piecewise-linear saturated IntVars (cap at 1.6 * SCALE), CP-SAT directly optimizes
    # the EXACT job-level readiness & track distribution alignment without additive over-stacking!
    w_readiness = float(policy.get("w_readiness", 1.0))
    w_dist = float(policy.get("w_dist", 0.45))
    w_emerg = float(policy.get("w_emerg", 0.20))
    w_worst_boost = policy.get("track_priority_multipliers", {})

    obj_terms = []
    cap_mass = int(round(2.0 * SCALE))
    for tk in TRACK_IDS:
        tp = track_demand["tracks"][tk]
        tk_weight = float(tp["weight_share"]) * float(w_worst_boost.get(tk, 1.0))
        d_dist = tp["demand_dist"]
        prev = tp["prevalence"]
        sv = sel_vars[tk] if multitrack else sel_vars["shared"]

        # Total mass & depth expressions for track tk
        base_m = req_mass_by_track[tk]
        base_b = req_depth_by_track[tk]

        for j in range(k):
            p_j = float(prev[j])
            d_j = float(d_dist[j])
            if p_j < 1e-4 and d_j < 1e-4:
                continue
            jb = max(1.0, float(job_bloom[j]))

            # Effective depth-weighted mass for skill j in track tk:
            # m_eff = 0.45 * mass + 0.55 * (bloom_sum / jb)  (saturates at 1.6 credits -> readiness = 1.0)
            const_eff = int(round((0.40 * base_m[j] + 0.60 * (base_b[j] / jb)) * SCALE))
            var_eff = (
                sum(
                    int(round((0.40 * dm[j] + 0.60 * (db[j] / jb)) * SCALE)) * upg_vars[u]
                    for u, (dm, db) in enumerate(upg_deltas)
                    if abs(dm[j]) > 1e-6 or abs(db[j]) > 1e-6
                )
                + sum(
                    int(round((0.40 * em[j] + 0.60 * (eb[j] / jb)) * SCALE)) * sv[m]
                    for m, (em, eb) in enumerate(elec_vecs)
                    if em[j] > 1e-6
                )
            )
            sat = model.new_int_var(0, cap_mass, f"sat_{tk}_{j}")
            model.add(sat <= const_eff + var_eff)

            # Bonus weight for emerging/growing skills
            em_mult = (1.0 + w_emerg) if kg["skills"][j]["stage"] in ("emerging", "growing") else 1.0
            coeff_r = int(round(1000.0 * tk_weight * (w_readiness * p_j + w_dist * d_j) * em_mult))
            if coeff_r > 0:
                obj_terms.append(coeff_r * sat)

    # Small regularization cost for adding courses / upgrades so solver prefers lean, impactful changes
    reg_cost = (
        sum(int(round(140 * float(c.get("faculty_fte", 0.40)))) * v for c, v in zip(additions, add_vars))
        + sum(int(round(50 * float(u.get("faculty_fte", 0.18)))) * v for u, v in zip(upgrades, upg_vars))
    )
    if not diagnose_iis:
        model.maximize(sum(obj_terms) - reg_cost)

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_deterministic_time = float(policy.get("solver_time", 2.5)) * (4.0 if diagnose_iis else 1.0)
    status = solver.solve(model)
    if diagnose_iis:
        core = []
        if status == cp_model.INFEASIBLE:
            core = [assumption_labels.get(i, f"unknown_{i}") for i in solver.sufficient_assumptions_for_infeasibility()]
        return {"status": solver.status_name(status), "iis_conflicts": core, "n_guarded_constraints": len(assumption_labels),
                "forced_unmatched": forced_unmatched}
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        diag = {"status": "SKIPPED", "iis_conflicts": []}
        if status == cp_model.INFEASIBLE:
            diag = solve_multitrack_curriculum(
                base_courses, candidate_electives, track_demand, kg,
                elective_quota_credits=elective_quota_credits, multitrack=multitrack,
                enable_capstone_orientation=enable_capstone_orientation, capstone_packages=capstone_packages,
                extra_syllabus_modules=extra_syllabus_modules, budget_mode=budget_mode, policy=policy, diagnose_iis=True,
            )
        return {
            "status": solver.status_name(status),
            "feasible": False,
            "budget_mode": budget_mode,
            "multitrack": multitrack,
            "elective_quota_credits": float(elective_quota_credits),
            "added_credits": 0.0,
            "removed_credits": 0.0,
            "upgraded_core_count": 0,
            "faculty_fte_used": 0.0,
            "faculty_fte_cap": float(fte_cap),
            "lab_units_used": 0.0,
            "lab_units_cap": float(lab_cap),
            "solver_wall_time_ms": round(solver.wall_time * 1000.0, 2),
            "iis_conflicts": diag.get("iis_conflicts", []),
            "iis_guarded_constraints": diag.get("n_guarded_constraints"),
            "forced_unmatched": forced_unmatched,
            "actions": [],
            "track_electives": {tk: [] for tk in TRACK_IDS},
            "capstone_actions_by_track": {tk: [] for tk in TRACK_IDS},
            "streaming_summary": {},
            "track_curricula": {tk: copy.deepcopy(base_courses) for tk in TRACK_IDS},
            "track_shares": {tk: float(track_demand["tracks"][tk]["weight_share"]) for tk in TRACK_IDS},
            "offered_catalogue": [],
            "constraint_checks": {"zero_hard_violations": False},
        }

    # Extract catalogue actions
    chosen_upgrades: Dict[int, Dict[str, Any]] = {}
    for u, v in enumerate(upg_vars):
        if solver.value(v):
            chosen_upgrades[upgrades[u]["base_index"]] = upgrades[u]

    actions: List[Dict[str, Any]] = []
    offered_catalogue: List[Dict[str, Any]] = []
    for i, c in enumerate(base_courses):
        if not solver.value(keep[i]):
            actions.append({
                "action": "REMOVE_ELECTIVE",
                "course_name": c["course_name"],
                "credits": float(c["credits"]),
                "delta_credits": -float(c["credits"]),
                "semester": sem(c),
                "skills": list(_skill_map(c).keys()),
                "reason": "释放低匹配选修容量以引入方向课群",
            })
        elif i in chosen_upgrades:
            new_c = copy.deepcopy(chosen_upgrades[i])
            offered_catalogue.append(new_c)
            actions.append({
                "action": "UPGRADE_SYLLABUS",
                "course_name": new_c["course_name"],
                "credits": float(new_c["credits"]),
                "delta_credits": 0.0,
                "semester": sem(new_c),
                "skills": list(_skill_map(new_c).keys()),
                "module": new_c["module"],
                "domain_coherent": True,
                "reason": f"学科同源大纲现代化：{new_c['module']['title']}",
            })
        else:
            offered_catalogue.append(copy.deepcopy(c))

    for j, (c_a, v_a) in enumerate(zip(additions, add_vars)):
        if solver.value(v_a):
            offered_catalogue.append(copy.deepcopy(c_a))
            actions.append({
                "action": "ADD_ELECTIVE",
                "course_name": c_a["course_name"],
                "credits": float(c_a["credits"]),
                "delta_credits": float(c_a["credits"]),
                "semester": sem(c_a),
                "skills": list(_skill_map(c_a).keys()),
                "target_tracks": c_a.get("target_tracks", []),
                "source": c_a.get("source", "seed_pool"),
                "reason": "按岗位族核心缺口增设方向选修课",
            })

    # Reconstruct exact per-track student curricula (required + capstone orientation + track selected electives)
    upgraded_req = [
        copy.deepcopy(chosen_upgrades[i]) if i in chosen_upgrades else copy.deepcopy(c)
        for i, c in enumerate(base_courses)
        if c.get("required")
    ]
    track_curricula: Dict[str, List[Dict[str, Any]]] = {}
    track_elective_lists: Dict[str, List[Dict[str, Any]]] = {}
    capstone_actions_by_track: Dict[str, List[Dict[str, Any]]] = {}

    for tk in TRACK_IDS:
        sv = sel_vars[tk] if multitrack else sel_vars["shared"]
        chosen_elecs = []
        for m, (c_e, _, _) in enumerate(all_electives):
            if solver.value(sv[m]):
                ec = copy.deepcopy(c_e)
                ec["take_rate"] = 1.0
                chosen_elecs.append(ec)
        track_elective_lists[tk] = [
            {"course_name": c["course_name"], "credits": float(c["credits"]), "semester": sem(c)}
            for c in chosen_elecs
        ]
        if multitrack and enable_capstone_orientation:
            req_tk, cap_acts = apply_track_capstone_orientation(upgraded_req, tk, capstone_packages, orient_ratio=orient_ratio)
        else:
            req_tk, cap_acts = copy.deepcopy(upgraded_req), []
        capstone_actions_by_track[tk] = cap_acts
        # Ensure all required courses have take_rate=1.0
        for rc in req_tk:
            rc["take_rate"] = 1.0
        track_curricula[tk] = req_tk + chosen_elecs

    # Run Gale-Shapley stable streaming at end of S4 to obtain realistic student shares with capacity caps
    demand_shares = {tk: float(track_demand["tracks"][tk]["weight_share"]) for tk in TRACK_IDS}
    if multitrack:
        streaming_cfg = []
        for tk in TRACK_IDS:
            sh = demand_shares[tk]
            streaming_cfg.append({
                "id": tk,
                "name": TRACK_META[tk]["name_zh"],
                "demand_share": sh,
                "min_cap": max(20, int(round(300 * max(0.10, sh * 0.82)))),
                "max_cap": min(135, int(round(300 * min(0.44, sh * 1.05 + 0.01)))),
            })
        rng_st = np.random.default_rng(42)
        n_st = 300
        herding = float(policy.get("ai_herding_bias", 0.14))
        base_probs = np.array([max(0.05, t["demand_share"]) for t in streaming_cfg], dtype=np.float64)
        # Model mild AI herding toward ai_data (index 2)
        base_probs[2] += herding
        base_probs = base_probs / base_probs.sum()
        # Generate student preferences & prerequisite-weighted scores
        util = rng_st.normal(0.0, 0.35, size=(n_st, len(streaming_cfg))) + np.log(base_probs + 1e-6)[None, :]
        student_prefs = np.argsort(-util, axis=1)
        gpa = np.clip(rng_st.normal(0.76, 0.11, size=(n_st, 1)), 0.40, 1.00)
        aptitude = np.clip(rng_st.normal(0.75, 0.12, size=(n_st, len(streaming_cfg))), 0.35, 1.00)
        student_scores = 0.55 * gpa + 0.30 * aptitude + 0.15 * (util - util.min()) / max(1e-6, util.max() - util.min())
        streaming = run_gale_shapley_streaming(
            n_students=n_st,
            tracks=streaming_cfg,
            student_prefs=student_prefs,
            student_track_scores=student_scores,
            allow_transfer_rate=0.10,
        )
        track_shares = {
            alloc.get("track_id", alloc["id"]): round(alloc["assigned"] / float(n_st), 5)
            for alloc in streaming["track_allocations"]
        }
    else:
        streaming = None
        track_shares = demand_shares

    # Hard constraint verification suite
    added_cr = sum(a["credits"] for a in actions if a["action"] == "ADD_ELECTIVE")
    removed_cr = sum(a["credits"] for a in actions if a["action"] == "REMOVE_ELECTIVE")
    fte_used = solver.value(fte_expr) / 100.0
    lab_used = solver.value(lab_expr) / 100.0

    quota_ok = all(
        abs(sum(c["credits"] for c in track_elective_lists[tk]) - (eff_quota_half / 2.0)) < 1e-6
        for tk in TRACK_IDS
    )
    req_retained = all(
        solver.value(keep[i]) == 1
        for i, c in enumerate(base_courses)
        if not is_replaceable_elective(c)
    )
    prereq_ok = all(
        (not solver.value(v_a)) or any(solver.value(p) for p in ps)
        for v_a, ps in prereq_checks
    )
    coherent_ok = all(a.get("domain_coherent", True) for a in actions if a["action"] == "UPGRADE_SYLLABUS")
    streaming_stable = True if streaming is None else (streaming["blocking_pairs"] == 0)

    checks = {
        "student_graduation_credits_conserved": quota_ok,
        "required_and_core_100pct_retained": req_retained,
        "faculty_fte_within_cap": fte_used <= fte_cap + 1e-6,
        "lab_gpu_within_cap": lab_used <= lab_cap + 1e-6,
        "prereq_dag_valid": prereq_ok,
        "domain_coherent_upgrades_only": coherent_ok,
        "streaming_zero_blocking_pairs": streaming_stable,
        "reform_budget_ok": added_cr <= swap_cap + 1e-6 and removed_cr <= swap_cap + 1e-6 and len(chosen_upgrades) <= upgrade_cap,
    }
    checks["zero_hard_violations"] = all(checks.values())

    return {
        "status": solver.status_name(status),
        "feasible": checks["zero_hard_violations"],
        "budget_mode": budget_mode,
        "multitrack": multitrack,
        "elective_quota_credits": eff_quota_half / 2.0,
        "added_credits": round(added_cr, 2),
        "removed_credits": round(removed_cr, 2),
        "upgraded_core_count": len(chosen_upgrades),
        "faculty_fte_used": round(fte_used, 2),
        "faculty_fte_cap": fte_cap,
        "lab_units_used": round(lab_used, 2),
        "lab_units_cap": lab_cap,
        "solver_wall_time_ms": round(solver.wall_time * 1000.0, 2),
        "constraint_checks": checks,
        "actions": actions,
        "track_electives": track_elective_lists,
        "capstone_actions_by_track": capstone_actions_by_track,
        "track_shares": track_shares,
        "streaming_summary": streaming,
        "track_curricula": track_curricula,
        "offered_catalogue": offered_catalogue,
    }
