"""Bounded and replayable outer search over teaching operators and demand scenarios."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import re

import numpy as np

from analytics.gap_engine import compute_supply_profile
from solver.curriculum_portfolio_solver import build_default_frontier_course_pool, solve_curriculum_cpsat
from solver.distribution_solver import near_duplicate

# These are proposed designs. Only the anchor course comes from the source curriculum.
RECIPES = (
    ('操作系统|Linux', 'linux_shell', '脚本化系统运维', '为给定服务编写启动、日志检查和异常恢复脚本', '隐藏测试验证正常启动、异常恢复和日志内容'),
    ('数据库', 'sql_rdbms', 'SQL性能与执行计划', '构建索引并比较同一数据集上的三种查询计划', '复现实验、结果正确性、分析报告各计分'),
    ('数据库', 'nosql_vector_db', '结构化与向量检索', '实现索引、检索和召回评测，比较结构化过滤与近邻查询', '报告召回率、延迟、查询局限并提交可复运行代码'),
    ('数据库|数据挖掘', 'big_data_eng', '批量数据处理', '用同一数据集实现分区计算，追踪数据缺失及重复记录', '测试数据完整性、重复处理和结果一致性'),
    ('人工智能|机器学习|数据挖掘', 'deep_learning', '神经网络训练与验证', '训练小型模型并比较基线、学习曲线和交叉验证误差', '提交可重复代码、实验矩阵和误差解释'),
    ('人工智能|数据挖掘|计算机导论', 'agent_rag', '检索增强应用评测', '在固定语料上做带引用的检索问答，比较无检索基线', '对检索召回、引文正确性和失败类型分别评分'),
    ('人工智能|计算机导论|数据挖掘', 'nlp_text', '文本分类与信息检索', '搭建词项检索与文本分类基线并做分层误差分析', '在独立样本检验检索与分类，提交误差归因'),
    ('程序设计|软件工程|大数据', 'python_eng', 'Python数据工程实践', '建立可复现的数据清洗、验证和批处理流水线', '检查异常输入、批处理结果与可复运行性'),
    ('软件工程|项目实训|面向对象', 'qa_testing', '自动化质量工程', '为现有课程项目编写单元、集成和回归测试', '用隐藏缺陷、边界用例和复现质量评分'),
    ('软件工程|项目实训|Java|面向对象', 'java_spring', '服务端应用设计', '实现数据访问、输入校验和有错误反馈的服务端应用', '验证接口契约、数据一致性与维护文档'),
    ('软件工程|Linux|计算机网络', 'devops_sre', '交付与系统可观测性', '为课程服务建立构建、部署和故障回滚流水线', '现场演示构建可重复性、故障定位和回滚'),
    ('组成原理|微机|操作系统', 'embedded_mcu', '嵌入式计算实验', '完成传感数据采集与周期任务调度实验', '检验时序、资源占用、测量误差和复验报告'),
    ('操作系统|计算机网络|软件工程', 'dist_systems', '分布式服务实验', '在受控实验中比较串行、并发及副本服务的一致性', '按复现实验、失败处理和性能归因评分'),
    ('智能计算系统|神经网络与深度学习', 'llm_posttrain', '预训练模型适配与评估', '对小型预训练模型进行参数高效适配并比较冻结基线', '检验测试集隔离、资源开销和泛化表现'),
    ('智能计算系统|边缘计算', 'cloud_k8s', '容器化模型服务', '把已有推理项目打包成可复用容器，执行资源和性能测试', '考核构建可复用性、资源配置和失败回退'),
    ('知识表示与推理|自然语言处理', 'agent_rag', '检索推理与证据追踪', '从固定文档检索支持证据，测试未覆盖问题上的拒答', '检验引用正确性、检索召回和边界样例'),
    ('AI新生工程体验|智能计算系统', 'qa_testing', '模型服务回归测试', '对给定预测服务编写数据契约与回归测试', '检验异常数据、版本差异与测试可重复性'),
    ('神经网络|机器学习|机器视觉', 'mlops_eval', '模型比较与监测', '基于学习曲线、分群指标及输入漂移比较两个候选模型', '检查基线、公平对比和对分群失效的解释'),
    ('Python语言程序设计|数据库|商务智能', 'sql_rdbms', '结构化数据查询与分析', '在现有课程项目上完成关系建模、联表、聚合与验收', '检查结果正确性、约束一致性和查询可复现性'),
    ('计算机技术|计算机组成|数字系统', 'linux_shell', 'Linux构建测试与实验自动化', '对现有课程实验编写编译、测试、复现实验的自动化脚本', '检验不同输入、日志以及命令执行可复现性'),
    ('计算机技术|智能计算系统|边缘计算', 'dist_systems', '分布式推理吞吐实验', '在受控教学集群比较单机和分片运行的性能', '考核吞吐记录、失败回退和瓶颈分析'),
    ('软件工程|项目实训|金融', 'hci_product', '行业项目需求与原型验证', '针对原教学场景编制用户需求、接口原型和验收用例', '按需求追踪、试用反馈和验收测试评分'),
)


def module_variants(base, skill_ids, fractions=(0.25,), recipes=RECIPES):
    """Reallocate at most 37.5% of one course to an assessed module."""
    variants = []
    for i, course in enumerate(base):
        if course.get('category') in ('general_edu', 'discipline_foundation') or course.get('practice_weeks', 0) > 0:
            continue
        hours = float(course.get('total_hours') or 0)
        lab = float(course.get('lab_hours') or 0)
        if (hours < 24 or not course.get('extracted_skills') or not 1 <= int(course.get('semester') or 0) <= 8
                or re.search(r'毕业|实习|思想|体育|金融|经济|专业导论', course['course_name'])):
            continue
        for pattern, skill, name, project, rubric in recipes:
            if skill not in skill_ids or not re.search(pattern, course['course_name'], re.I):
                continue
            for fraction in fractions:
                module_hours = int(min(hours * fraction, 16) // 2 * 2)
                share = module_hours / hours
                if module_hours < 6:
                    continue
                prev = {sk['id']: sk for sk in course['extracted_skills']}
                if skill in prev and (prev[skill]['weight'] > 0.7 or prev[skill]['bloom'] >= 4 and prev[skill]['weight'] > 0.45):
                    continue
                weights = {sid: sk['weight'] * (1-share) for sid, sk in prev.items()}
                weights[skill] = weights.get(skill, 0) + share
                depth = {sid: float(sk['bloom']) for sid, sk in prev.items()}
                depth[skill] = ((weights[skill]-share)*depth.get(skill, 0) + share*3.0)/weights[skill]
                edited = dict(course, base_index=i, action_type='UPGRADE_SYLLABUS', original_syllabus=deepcopy(course),
                              extracted_skills=[{'id': sid, 'weight': w, 'bloom': depth[sid]} for sid, w in weights.items()],
                              faculty_fte=round(0.08 + module_hours/160, 2), lab_units=round(module_hours/32, 2),
                              lab_hours=min(hours, max(lab, module_hours)),
                              module={'title': name, 'instruction_hours': module_hours, 'assessment': rubric,
                                      'student_project': project, 'planned_bloom': 3,
                                      'evidence': {'course_code': course.get('course_code'), 'page': course.get('source_page'),
                                                   'program_id': course.get('program_id'), 'official_syllabus_available': bool(course.get('has_syllabus'))}})
                variants.append(edited)
    return variants


def terminology_candidates(train, kg, train_years=None):
    """Rank unlisted atoms by contrastive association with an anchored skill.

    These are candidate aliases for review. Scores remain in the frozen ontology.
    """
    aliases = {term.lower() for sk in kg['skills'] for term in sk.get('synonyms', [])}
    incidence = defaultdict(list)
    hits = train['hits']
    marginal = hits.mean(axis=0)
    for i, terms in enumerate(train['atomic_terms']):
        for term in {str(t).lower() for t in terms} - aliases:
            if re.fullmatch(r'[a-z][a-z0-9.+_-]{2,24}', term):
                incidence[term].append(i)
    ranked = []
    for term, rows in incidence.items():
        n = len(rows)
        employers = len({train['employers'][i] for i in rows})
        if n < 12 or employers < 6:
            continue
        rate = hits[rows].mean(axis=0)
        index = int(np.argmax(rate - marginal))
        if rate[index] < 0.65 or rate[index] - marginal[index] < 0.45:
            continue
        if train_years is not None:
            hold = incidence_by_term(train_years, term)
            if len(hold) < 4 or train_years['hits'][hold, index].mean() < 0.60:
                continue
        ranked.append({'term': term, 'skill_id': kg['skill_ids'][index], 'ads': n, 'employers': employers,
                       'conditional_frequency': round(float(rate[index]), 4)})
    return sorted(ranked, key=lambda r: (-r['ads'], r['term']))[:12]


def incidence_by_term(demand, term):
    return [i for i, terms in enumerate(demand['atomic_terms']) if any(str(t).lower() == term for t in terms)]


def normalized_metrics(courses, demand, kg):
    supply = compute_supply_profile(courses, kg['skill_ids'])
    weight = demand['demand_dist']
    depth = supply['bloom'] * np.minimum(1.0, supply['mass']/2.0)
    readiness = np.minimum(1.0, depth / np.maximum(1, demand['job_bloom']))
    overlap = float(np.minimum(weight, supply['dist']).sum())
    # Averaging each skill fraction first yields simple per-JD skill coverage.
    hits = demand['hits']
    count = hits.sum(axis=1)
    mask = count > 0
    coverage = (hits[mask] @ readiness) / count[mask]
    job_coverage = float(np.average(coverage, weights=demand['ad_weights'][mask]))
    return {'l1_overlap': overlap, 'jd_skill_coverage': job_coverage,
            'mapped_credits': float(supply['mass'].sum())}


def objective(metrics):
    return 0.65 * metrics['l1_overlap'] + 0.35 * metrics['jd_skill_coverage']


def _novel(pool, additions):
    checked = list(pool)
    for course in additions:
        if not any(near_duplicate(course, old) for old in checked if old.get('action_type') != 'UPGRADE_SYLLABUS'):
            checked.append(course)
    return checked


def evolve_structure(base, train, recent_train, dev, kg, *, llm=None, generations=3, budget_mode='medium'):
    """Score all structural variants with the same validation objective and store failures."""
    seed = build_default_frontier_course_pool()
    candidate_pool = list(seed)
    authored_modules = []
    timeline = []
    champion = None
    champion_metrics = None
    champion_spec = None
    aliases = terminology_candidates(train, kg, recent_train)
    # This branch only affects planning. Benchmark features use fixed skills and aliases.
    counts = train['hits'].copy()
    skill_index = {sid: i for i, sid in enumerate(kg['skill_ids'])}
    for record in aliases:
        j = skill_index[record['skill_id']]
        counts[incidence_by_term(train, record['term']), j] = 1
    changed = (counts * train['ad_weights'][:, None]).sum(axis=0)
    changed = changed / changed.sum()
    structural = (
        ('static_catalogue', (), (), {}),
        ('protected_syllabus_modules', (0.25,), (), {}),
        ('expanded_module_sizes', (0.25, 0.375), (), {}),
        ('multi_year_robust_modules', (0.25, 0.375), (recent_train['demand_dist'],), {}),
        ('evidence_alias_planning', (0.25,), (changed, recent_train['demand_dist']), {}),
        ('job_readiness_modules', (0.25, 0.375), (recent_train['demand_dist'],), {'jd_depth_weight': 1.0}),
        ('occupational_gap_modules', (0.25, 0.375), (), {'jd_depth_weight': 0.65}),
        ('recent_market_focus', (0.25, 0.375), (), {'fit_population': 'recent'}),
        ('demand_adapted_mix', (0.25, 0.375), (train['demand_dist'],), {'fit_population': 'recent'}),
    )
    for gen in range(generations + 1):
        if llm is not None and gen:
            # Select targets using fit-period deficits in the currently accepted plan.
            current = champion['reconstructed_courses'] if champion else base
            mass = compute_supply_profile(current, kg['skill_ids'])['dist']
            deficit = 0.60 * recent_train['demand_dist'] + 0.40 * train['demand_dist'] - mass
            metrics_for_proposer = {**(champion_metrics or {}), 'fit_skill_gaps': [
                {'skill_id': kg['skill_ids'][j], 'gap': round(float(deficit[j]), 5)}
                for j in np.argsort(deficit)[-10:][::-1] if deficit[j] > 0]}
            generated = llm(gen, metrics_for_proposer, timeline, base, kg)
            candidate_pool = _novel(candidate_pool, [c for c in generated if c['action_type'] == 'ADD_COURSE'])
            for revision in (c for c in generated if c['action_type'] == 'UPGRADE_SYLLABUS'):
                key = (revision['base_index'], tuple(s['id'] for s in revision['extracted_skills']))
                if not any(key == (c['base_index'], tuple(s['id'] for s in c['extracted_skills'])) for c in authored_modules):
                    authored_modules.append(revision)
        if gen == 0:
            specs = structural[:1]
        else:
            tried = {h['spec'] for h in timeline if h['generation'] == gen-1 and not h['accepted']}
            specs = [s for s in structural[1:] if s[0] not in tried] or list(structural[1:])
            if champion_spec and champion_spec[0] not in {s[0] for s in specs}:
                specs.append(champion_spec)
        round_results = []
        for spec in specs:
            label, fractions, scenarios, policy = spec
            pool = candidate_pool + module_variants(base, kg['skill_ids'], fractions) + (authored_modules if fractions else [])
            fit_demand = recent_train if policy.get('fit_population') == 'recent' else train
            solution = solve_curriculum_cpsat(base, pool, fit_demand, kg, budget_mode=budget_mode,
                                               policy={**policy, 'scenarios': scenarios})
            if not solution['feasible']:
                timeline.append({'generation': gen, 'spec': label, 'accepted': False, 'reason': solution['status']})
                continue
            metrics = normalized_metrics(solution['reconstructed_courses'], dev, kg)
            recent_metrics = normalized_metrics(solution['reconstructed_courses'], recent_train, kg)
            round_results.append((objective(metrics), spec, solution, metrics, recent_metrics))
        selection = None
        for ranked in sorted(round_results, key=lambda row: row[0], reverse=True):
            score, spec, solution, metrics, recent_metrics = ranked
            if champion is None or (score > objective(champion_metrics)+0.001 and
                    metrics['l1_overlap'] >= champion_metrics['l1_overlap']-0.003 and
                    recent_metrics['l1_overlap'] >= normalized_metrics(champion['reconstructed_courses'], recent_train, kg)['l1_overlap']-0.005):
                selection = ranked
                break
        for ranked in round_results:
            score, spec, solution, metrics, recent_metrics = ranked
            promoted = ranked is selection
            timeline.append({'generation': gen, 'spec': spec[0], 'score': score, 'accepted': promoted,
                             'metrics': metrics, 'recent_train_metrics': recent_metrics,
                             'selected_actions': solution['actions'],
                             'reason': 'promoted' if promoted else 'lower_score_or_regression',
                             'pool_size': len(candidate_pool)})
        if selection is not None:
            _, champion_spec, champion, champion_metrics, _ = selection
    if champion is None:
        raise RuntimeError('No feasible harness state')
    return {'solution': champion, 'selected_spec': champion_spec[0], 'timeline': timeline,
            'aliases': aliases, 'llm_candidates': candidate_pool[len(seed):], 'authored_modules': authored_modules}
