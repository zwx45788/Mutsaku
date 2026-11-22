"""Re-evaluate the four repairs without upgrading the existing delivery artifacts in place."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import re

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analytics.gap_engine import compute_supply_profile, exact_wasserstein_1, extract_demand_profile, load_curricula, load_kg
from analytics.pathways import expected_pathway, group_curricula
from llm.client import QwenClient
from rsi.harness_rsi_engine import canonical_sha256
from rsi.structural_search import evolve_structure, module_variants, normalized_metrics
from security.leakage_guard import BlindAuditContext
from simulator.dream_sim import fit_conformal_salary_predictor, run_dream_sim_4year
from solver.curriculum_portfolio_solver import build_default_frontier_course_pool, solve_additive_legacy, solve_curriculum_cpsat
from solver.elective_path import realize_path, recommend_elective_path

ROOT = Path('evoalign_os')
PROC = ROOT / 'data/processed'
HOLD = ROOT / 'data/blind_holdout'
OUT = ROOT / 'outputs'
PROGRAMS = ('BTBU_CS_2022', 'BTBU_AI_2022', 'BISTU_CS_2025')


def model_proposer(audit):
    client = QwenClient(model='qwen3.8-flash', timeout=30.0)

    def propose(generation, current, previous, base, kg):
        try:
            response = client.chat_json(
                system='为高校提议互补的计算机课程草案，用给定技能ID。只输出JSON对象，含courses列表。每项含name, skill_ids（最多3个）, prerequisite（一个基座技能ID）, project, assessment。给出可评估的本科项目和验收标准。另输出 modules 列表，每项含anchor_code（给定现有课代码）、target_skill（一个缺口技能ID）、name、project、assessment。模块必须在已有课程内完成，保留主干内容并能在最多16学时内验收。',
                user=json.dumps({'generation': generation, 'metrics': current,
                                 'program': base[0]['program_id'],
                                 'recent_rejections': [(h['spec'], h['reason']) for h in previous if not h['accepted']][-6:],
                                 'anchor_courses': [{'code': c.get('course_code'), 'name': c['course_name'], 'semester': c.get('semester'),
                                                     'skills': [s['id'] for s in c.get('extracted_skills', ())]}
                                                    for c in base if c.get('category') != 'general_edu' and c.get('semester')],
                                 'skills': {s['id']: s['zh'] for s in kg['skills']}}, ensure_ascii=False),
                tag=f'four_repairs_{base[0]["program_id"]}_{generation}', max_tokens=900, enable_thinking=False,
            )
        except Exception as error:
            audit.append({'generation': generation, 'error_type': type(error).__name__})
            return []
        audit.append({'generation': generation, **response['meta']})
        skill_ids = set(kg['skill_ids'])
        courses = []
        if not isinstance(response['data'], dict):
            return courses
        for item in response['data'].get('courses', [])[:2]:
            if not isinstance(item, dict):
                continue
            name = str(item.get('name', '')).strip()
            project, assessment = str(item.get('project', '')), str(item.get('assessment', ''))
            prerequisite = item.get('prerequisite')
            ids = list(dict.fromkeys(s for s in item.get('skill_ids', []) if s in skill_ids))[:3]
            if not (4 <= len(name) <= 45 and len(project) >= 12 and len(assessment) >= 8 and ids and prerequisite in skill_ids):
                continue
            if re.search('攻击|攻防|漏洞利用|渗透', name + project + assessment):
                continue
            courses.append({'course_id': canonical_sha256(name)[:12], 'action_type': 'ADD_COURSE', 'course_name': name,
                            'credits': 2.0, 'total_hours': 32.0, 'lab_hours': 16.0, 'practice_weeks': 0.0,
                            'semester': 6, 'category': 'major_elective', 'required': False, 'faculty_fte': 0.4,
                            'lab_units': 1.0, 'extracted_skills': [{'id': s, 'weight': 1/len(ids), 'bloom': 3} for s in ids],
                            'prereq_skills': [prerequisite], 'module': {'student_project': project, 'assessment': assessment}})
        anchors = {c.get('course_code'): c for c in base if c.get('course_code') and c.get('extracted_skills')}
        domains = {s['id']: s['domain'] for s in kg['skills']}
        connections = {(edge['from'], edge['to']) for edge in kg['skill_prereq_edges']}
        for item in response['data'].get('modules', [])[:4]:
            if not isinstance(item, dict):
                continue
            anchor = anchors.get(item.get('anchor_code'))
            target = item.get('target_skill')
            title = str(item.get('name', '')).strip()
            project, assessment = str(item.get('project', '')), str(item.get('assessment', ''))
            if not (anchor and target in domains and 4 <= len(title) <= 45 and 12 <= len(project) <= 240 and 8 <= len(assessment) <= 240):
                continue
            if re.search('攻击|攻防|漏洞利用|渗透', title + project + assessment):
                continue
            anchor_skills = {s['id'] for s in anchor['extracted_skills']}
            # Require documented vocabulary continuity or a prerequisite link at the anchor.
            if not any(domains.get(s) == domains[target] or (s, target) in connections for s in anchor_skills):
                continue
            recipe = (re.escape(anchor['course_name']), target, title, project, assessment)
            revisions = module_variants(base, kg['skill_ids'], (0.25, 0.375), recipes=(recipe,))
            for revision in revisions:
                if revision.get('course_code') == anchor['course_code']:
                    revision['module']['qwen_request_hash'] = response['meta']['request_hash']
                    courses.append(revision)
        return courses
    return propose


def paired_overlap(profiles, reference, demand, n_boot=600, seed=45):
    employers, group = np.unique(demand['employers'], return_inverse=True)
    employer_features = np.zeros((len(employers), demand['hits'].shape[1]))
    features = demand['hits'] * demand['ad_weights'][:, None]
    np.add.at(employer_features, group, features)
    # Use the same data/weight normalization as reported L1 scores, including any fixed policy tilt.
    scaling = demand['demand_dist'] / np.maximum(employer_features.sum(axis=0), 1e-13)
    employer_features *= scaling
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(len(employers), np.full(len(employers), 1/len(employers)), size=n_boot)
    draws = draws @ employer_features
    draws /= np.maximum(draws.sum(axis=1, keepdims=True), 1e-13)
    ref = np.minimum(draws, reference).sum(axis=1)
    return {key: [round(float(x), 6) for x in np.quantile(np.minimum(draws, profile).sum(axis=1)-ref, (0.025, 0.975))]
            for key, profile in profiles.items()}


def brief(solution):
    return {k: v for k, v in solution.items() if k not in ('reconstructed_courses', 'solver_wall_time_ms')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--qwen', action='store_true')
    parser.add_argument('--dev-only', action='store_true')
    args = parser.parse_args()
    kg = load_kg()
    groups = group_curricula(load_curricula())
    train_file = PROC / 'demand_train_2018_2023.parquet'
    train = extract_demand_profile([train_file], kg, min_year=2022, max_year=2023, use_bj_shortage_calibration=False)
    recent = extract_demand_profile([train_file], kg, min_year=2023, max_year=2023, use_bj_shortage_calibration=False)
    dev = extract_demand_profile([PROC / 'demand_dev_2024.parquet'], kg, use_bj_shortage_calibration=False)
    conf = fit_conformal_salary_predictor(train, dev)
    costs = np.asarray(kg['semantic_ground_distance']['matrix'])
    all_cases = {}
    replay = {}
    for program in PROGRAMS:
        print(f'Planning {program}', flush=True)
        base, pathway = expected_pathway(groups[program])
        pool = build_default_frontier_course_pool()
        modules = module_variants(base, kg['skill_ids'], (0.25, 0.375))
        model_audit = []
        proposer = model_proposer(model_audit) if args.qwen else None
        evolved = evolve_structure(base, train, recent, dev, kg, llm=proposer)
        identical_pool = pool + evolved['llm_candidates'] + modules + evolved['authored_modules']
        plans = {
            'freeze': {'reconstructed_courses': base, 'actions': []},
            'legacy_additive': solve_additive_legacy(base, pool, train, kg),
            'static_distribution': solve_curriculum_cpsat(base, pool, train, kg),
            'protected_additive': solve_curriculum_cpsat(base, identical_pool, train, kg, policy={'objective': 'additive'}),
            'static_modules': solve_curriculum_cpsat(base, identical_pool, train, kg),
            'adaptive_structure': evolved['solution'],
        }
        for method, plan in plans.items():
            if method != 'freeze' and not plan['feasible']:
                raise RuntimeError(f'{program}/{method}: {plan["status"]}')
        # Evaluate an executable elective recommendation at the same credits and assumed adoption.
        repaired = plans['adaptive_structure']['reconstructed_courses']
        quota = round(2*sum(c['credits']*c['take_rate'] for c in repaired
                            if c.get('category') == 'major_elective' and not c.get('required')))/2
        elective_paths = {}
        for criteria in ('additive', 'distribution'):
            path = recommend_elective_path(repaired, train['demand_dist'], kg['skill_ids'],
                                           quota_credits=quota, objective=criteria)
            elective_paths[criteria] = path
            if path['status'] in ('OPTIMAL', 'FEASIBLE'):
                for uptake in (0.25, 0.50, 0.75):
                    method = f'path_{criteria}_{round(100*uptake)}'
                    plans[method] = {'reconstructed_courses': realize_path(repaired, path['selected'], uptake),
                                     'actions': [], 'uptake': uptake}
        joint = solve_curriculum_cpsat(base, identical_pool, train, kg,
                                      policy={'path_quota': quota, 'path_uptake': 0.50})
        if joint['feasible']:
            joint_catalogue = joint['reconstructed_courses']
            chosen = joint['path_selected']
            elective_paths['joint'] = {'status': joint['status'], 'quota_credits': quota,
                                      'courses': [{'name': joint_catalogue[i]['course_name'],
                                                   'credits': joint_catalogue[i]['credits'],
                                                   'semester': joint_catalogue[i]['semester']} for i in chosen]}
            plans['joint_distribution_50'] = {**joint, 'uptake': 0.50,
                'reconstructed_courses': realize_path(joint_catalogue, chosen, 0.50)}
        else:
            elective_paths['joint'] = {'status': joint['status'], 'iis_conflicts': joint['iis_conflicts']}
        # Select architecture before opening either evaluation slice. Conditional on the same 50% uptake.
        shortlist = [m for m in ('path_distribution_50', 'joint_distribution_50') if m in plans]
        recommendation = max(shortlist, key=lambda m: normalized_metrics(plans[m]['reconstructed_courses'], dev, kg)['l1_overlap']) if shortlist else 'adaptive_structure'
        all_cases[program] = plans
        replay[program] = {'pathway': pathway, 'actions': {k: brief(v) for k, v in plans.items()},
                           'selected_spec': evolved['selected_spec'], 'timeline': evolved['timeline'],
                           'elective_paths': elective_paths, 'recommended_by_dev_l1': recommendation,
                           'alias_evidence': evolved['aliases'], 'qwen_calls': model_audit, 'results': {}}
        print(program, evolved['selected_spec'], {k: round(normalized_metrics(v['reconstructed_courses'], dev, kg)['l1_overlap'], 4) for k,v in plans.items()}, flush=True)
    snapshot = {p: {m: canonical_sha256(s['reconstructed_courses']) for m,s in plans.items()} for p,plans in all_cases.items()}
    demand_sets = {'development_2024': dev}
    if not args.dev_only:
        with BlindAuditContext('locked structural repair replay; previously inspected dates', caller='run_structural_repair'):
            for year, file in [('replay_2025', 'demand_holdout_2025.parquet'), ('replay_2026q1', 'demand_drift_2026q1.parquet')]:
                demand_sets[year] = extract_demand_profile([HOLD / file], kg, use_bj_shortage_calibration=False)
    for year, demand in demand_sets.items():
        for program, plans in all_cases.items():
            distributions, results = {}, {}
            for key, plan in plans.items():
                curriculum = plan['reconstructed_courses']
                distributions[key] = compute_supply_profile(curriculum, kg['skill_ids'])['dist']
                metrics = normalized_metrics(curriculum, demand, kg)
                metrics['ot_alignment'] = 1 - exact_wasserstein_1(distributions[key], demand['demand_dist'], costs)
                sim = run_dream_sim_4year(curriculum, demand, kg, conf, n_students=240)
                metrics['sim_jd_coverage'] = sim['skill_qualification_proxy_rate']
                results[key] = metrics
            paired = {reference: paired_overlap(distributions, distributions[reference], demand)
                      for reference in ('freeze', 'legacy_additive', 'protected_additive', 'static_modules',
                                        'adaptive_structure', 'path_additive_50', 'path_distribution_50') if reference in distributions}
            replay[program]['results'][year] = {'ads': demand['n_rows'], 'employers': demand['n_employers'],
                                                 'metrics': results, 'paired_overlap_ci95': paired}
    result = {'schema': 'structural-repair-v1', 'design': {'development': '2024',
              'population': 'broad_entry on every date', 'credit_budget_primary': 'official degree tables',
              'holdout_previously_seen_in_prior_runs': True,
              'primary_metric': 'L1 overlap with employer-balanced JD mentions', 'resamples': 600,
              'plan_selection': 'max development-2024 L1 among feasible distribution paths at 50% uptake',
              'minimum_practical_gain_pp': 2, 'qwen_enabled': args.qwen,
              'elective_uptake_scenarios': [0.25, 0.50, 0.75]}, 'plan_hashes': snapshot, 'cases': replay}
    result['sha256'] = canonical_sha256(result)
    out = OUT / ('structural_repair_dev.json' if args.dev_only else 'structural_repair_results.json')
    temp = out.with_suffix('.tmp')
    temp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(out)
    print(f'Results written to {out}', flush=True)


if __name__ == '__main__':
    main()
