"""Credit-conserving distribution optimization with explicit syllabus and safety checks."""
from __future__ import annotations

import copy
import re
import unicodedata

import numpy as np
from ortools.sat.python import cp_model

CAPS = {"lite": (8, 2.0, 6, 3), "medium": (18, 4.2, 13, 6), "heavy": (30, 6.8, 20, 10)}
SCALE = 1000
PROTECTED = re.compile(r"导论|安全|数学|代数|概率论|体育|身体素质|体质|军训|军理|思想|马克思|习近平|毛泽东|实习|毕业|经济|金融|创新创业|劳动|形势与政策|英语|English|国防|心理健康", re.I)


def course_mass(course, index):
    """Expected skill credits and Bloom-equivalent credits per course."""
    mass = np.zeros(len(index))
    depth = np.zeros_like(mass)
    if course.get("category") == "general_edu":
        return mass, depth
    credits = float(course["credits"])
    exposure = float(course.get("take_rate", 1.0 if course.get("required") else 0.55))
    for skill in course.get("extracted_skills", ()):
        sid = skill["id"]
        if sid in index:
            j = index[sid]
            weight = credits * exposure * float(skill["weight"])
            mass[j] += weight
            depth[j] += weight * float(skill["bloom"])
    return mass, depth


def _name(value):
    return ''.join(c for c in unicodedata.normalize('NFKC', value).lower() if c.isalnum())


def _grams(text):
    return {text[i:i+2] for i in range(len(text) - 1)}


def _skills(row):
    return {s['id']: float(s['weight']) for s in row.get('extracted_skills', []) if s['weight'] > 0}


def near_duplicate(a, b, threshold=0.70):
    an, bn = _name(a['course_name']), _name(b['course_name'])
    if an == bn:
        return True
    x, y = _skills(a), _skills(b)
    if not x or not y:
        return False
    overlap = sum(min(w, y.get(s, 0)) for s, w in x.items())
    overlap /= min(sum(x.values()), sum(y.values()))
    grams_a, grams_b = _grams(an), _grams(bn)
    lexical = len(grams_a & grams_b) / max(1, min(len(grams_a), len(grams_b)))
    return (overlap >= threshold and lexical >= 0.25) or (overlap >= 0.88 and set(x) == set(y))


def solve_distribution(base, candidates, demand, kg, *, budget_mode='medium', policy=None, strict=True):
    policy = policy or {}
    changed_cap, fte_cap, lab_cap, upgrades_cap = CAPS.get(budget_mode, CAPS['medium'])
    index = {sid: i for i, sid in enumerate(kg['skill_ids'])}
    model = cp_model.CpModel()
    labels = {}

    def constrain(condition, name):
        assumption = model.new_bool_var(name)
        model.add(condition).only_enforce_if(assumption)
        model.add_assumption(assumption)
        labels[assumption.index] = name

    def locked(c):
        return (c.get('required') or c.get('category') != 'major_elective' or c.get('term_evidence') == 'unverified'
                or c.get('is_core_foundation') or c.get('practice_weeks', 0) > 0 or PROTECTED.search(c['course_name']))

    def half(c):
        return round(float(c['credits']) * 2)

    def sem(c):
        return int(c.get('semester') or 0)

    keep = [model.new_constant(1) if locked(c) else model.new_bool_var(f'keep_{i}') for i, c in enumerate(base)]
    additions, revisions = [], []
    for row in candidates:
        c = copy.deepcopy(row)
        if c.get('action_type') == 'UPGRADE_SYLLABUS':
            i = c.get('base_index')
            if not isinstance(i, int) or not 0 <= i < len(base) or not c.get('module'):
                continue
            old = base[i]
            if (c.get('course_name') != old['course_name'] or half(c) != half(old) or sem(c) != sem(old)
                    or c.get('total_hours') != old.get('total_hours') or c.get('practice_weeks') != old.get('practice_weeks')):
                continue
            revisions.append((i, c, model.new_bool_var(f'revise_{len(revisions)}')))
        elif c.get('credits', 0) > 0 and 1 <= sem(c) <= 8 and _skills(c):
            c['action_type'] = 'ADD_COURSE'
            c['required'] = False
            c['catalogue_subtype'] = c.get('category', 'major_elective')
            c['category'] = 'major_elective'
            c['take_rate'] = next((float(x['take_rate']) for x in base
                                   if x.get('category') == 'major_elective' and not x.get('required') and 'take_rate' in x), 0.55)
            additions.append((c, model.new_bool_var(f'add_{len(additions)}')))
    for title in sorted({c["module"]["title"] for _, c, _ in revisions if "title" in c["module"]}):
        constrain(sum(v for _, c, v in revisions if c["module"].get("title") == title) <= 1, f"module_repeat_{title}")
    for i in {i for i, _, _ in revisions}:
        constrain(sum(v for j, _, v in revisions if j == i) <= keep[i], f'at_most_one_revision_{i}')
    add_cr = sum(half(c) * v for c, v in additions)
    drop_cr = sum(half(c) * (1 - v) for c, v in zip(base, keep))
    slack = 0 if strict else 2
    constrain(add_cr - drop_cr <= slack, 'credit_balance_upper')
    constrain(add_cr - drop_cr >= -slack, 'credit_balance_lower')
    constrain(add_cr <= 2 * changed_cap, 'added_credit_cap')
    constrain(drop_cr <= 2 * changed_cap, 'removed_credit_cap')
    constrain(sum(v for _, _, v in revisions) <= upgrades_cap, 'revision_cap')
    edits = additions + [(c, v) for _, c, v in revisions]
    fte = sum(round(c.get('faculty_fte', 0.45)*100) * v for c, v in edits)
    lab = sum(round(c.get('lab_units', 1.2)*100) * v for c, v in edits)
    constrain(fte <= round(fte_cap*100), 'faculty_cap')
    constrain(lab <= round(lab_cap*100), 'lab_cap')
    for s in range(1, 9):
        base_load = sum(half(c) for c in base if sem(c) == s)
        load = sum(half(c)*v for c, v in zip(base, keep) if sem(c) == s)
        load += sum(half(c)*v for c, v in additions if sem(c) == s)
        constrain(load <= max(base_load, 56), f'semester_{s}_credit_cap')

    threshold = float(policy.get('duplicate_threshold', 0.70))
    conflicts = []
    for j, (a, v) in enumerate(additions):
        for i, (b, k) in enumerate(list(zip(base, keep)) + additions[:j]):
            if near_duplicate(a, b, threshold):
                conflicts.append((v, k))
                constrain(v + k <= 1, f'duplicate_{j}_{i}')
    prerequisite_checks = []
    for i, (a, v) in enumerate(additions):
        for sid in sorted(set(a.get('prereq_skills', ()))):
            pairs = list(zip(base, keep)) + additions
            providers = [k for b, k in pairs if 0 < sem(b) < sem(a) and _skills(b).get(sid, 0) >= 0.10]
            prerequisite_checks.append((v, providers))
            constrain(sum(providers) >= v, f'prerequisite_{i}_{sid}')

    # Correct for endogenous normalization. For each skill, M_j / sum(M) - D_j
    # has numerator M_j - D_j * sum(M). Credit-denominated loss keeps the MILP linear.
    base_vectors = [course_mass(c, index) for c in base]
    vectors = [(v, m, b) for v, (m, b) in zip(keep, base_vectors)]
    vectors += [(v, *course_mass(c, index)) for c, v in additions]
    for i, c, v in revisions:
        m, b = course_mass(c, index)
        old_m, old_b = base_vectors[i]
        vectors.append((v, m - old_m, b - old_b))
    catalogue_vectors = list(vectors)
    selection = {}
    selected_requirements = []
    uptake = float(policy.get('path_uptake', 0.5))
    quota = policy.get('path_quota')
    joint = quota is not None
    if joint:
        if not 0 < uptake <= 1 or quota < 0 or abs(2*quota - round(2*quota)) > 1e-8:
            raise ValueError('A joint path requires positive uptake and a half-credit quota')
        catalogue = list(zip(base, keep)) + additions
        for i, (course, offered) in enumerate(catalogue):
            if course.get('required') or course.get('category') != 'major_elective':
                continue
            selected = model.new_bool_var(f'path_{i}')
            selection[i] = selected
            model.add(selected <= offered)
            if not 1 <= sem(course) <= 8 or course.get('term_evidence') == 'unverified':
                model.add(selected == 0)
        constrain(sum(half(catalogue[i][0])*v for i, v in selection.items()) == round(2*quota), 'path_credit_quota')
        for s in range(1, 9):
            required_load = sum(half(c) for c in base if c.get('required') and sem(c) == s)
            constrain(sum(half(catalogue[i][0])*v for i, v in selection.items() if sem(catalogue[i][0]) == s)
                      <= max(0, 56-required_load), f'path_semester_{s}_cap')
        for i, selected in selection.items():
            course = catalogue[i][0]
            for j, other_selected in selection.items():
                if j < i and near_duplicate(course, catalogue[j][0], threshold):
                    constrain(selected + other_selected <= 1, f'path_duplicate_{i}_{j}')
            for sid in sorted(set(course.get('prereq_skills', ()))):
                providers = [v for c, v in catalogue if c.get('required') and 0 < sem(c) < sem(course)
                             and _skills(c).get(sid, 0) >= 0.10]
                providers += [v for j, v in selection.items() if 0 < sem(catalogue[j][0]) < sem(course)
                              and _skills(catalogue[j][0]).get(sid, 0) >= 0.10]
                selected_requirements.append((selected, providers))
                constrain(sum(providers) >= selected, f'path_prerequisite_{i}_{sid}')
            mass, depth = course_mass(dict(course, take_rate=1.0), index)
            offered, expected_mass, expected_depth = vectors[i]
            vectors[i] = (offered, expected_mass*(1-uptake), expected_depth*(1-uptake))
            vectors.append((selected, mass*uptake, depth*uptake))
        for j, (i, course, revise) in enumerate(revisions):
            if i not in selection:
                continue
            revision_index = len(catalogue) + j
            _, mass, depth = vectors[revision_index]
            vectors[revision_index] = (revise, mass*(1-uptake), depth*(1-uptake))
            selected_revision = model.new_bool_var(f'path_with_revision_{j}')
            model.add_multiplication_equality(selected_revision, [revise, selection[i]])
            old_mass, old_depth = course_mass(dict(base[i], take_rate=1.0), index)
            new_mass, new_depth = course_mass(dict(course, take_rate=1.0), index)
            vectors.append((selected_revision, uptake*(new_mass-old_mass), uptake*(new_depth-old_depth)))
    mass_coeff = [np.rint(m * SCALE).astype(int) for _, m, _ in vectors]
    depth_coeff = [np.rint(b * SCALE).astype(int) for _, _, b in vectors]
    total_coeff = [int(m.sum()) for m in mass_coeff]
    for j, skill in enumerate(kg['skills']):
        if not skill.get('core'):
            continue
        base_mass = sum(round(float(m[j]) * SCALE) for m, _ in base_vectors)
        preserved = sum(round(float(m[j]) * SCALE) * v for v, m, _ in catalogue_vectors)
        minimum = round(base_mass * 0.80)
        constrain(preserved >= minimum, f'core_exposure_{skill["id"]}')
        if joint:
            realized = sum(int(m[j])*v for (v, _, _), m in zip(vectors, mass_coeff))
            constrain(realized >= minimum, f'path_core_exposure_{skill["id"]}')
    bound = max(1, 10 * (sum(abs(x) for x in total_coeff) + int(sum(abs(m).sum() for m in mass_coeff))))
    primary = np.asarray(demand['demand_dist'])
    depth_target = np.asarray(demand['job_bloom'])
    scenarios = policy.get('scenarios', ())
    profiles = [(primary, 100)] if not scenarios else [(primary, 80)] + [(p, 20 // len(scenarios)) for p in scenarios]
    objective = []
    def numerator(j, p):
        return sum(int(m[j] - round(t * p)) * v for (v, _, _), m, t in zip(vectors, mass_coeff, total_coeff))

    for s, (profile, weight) in enumerate(profiles):
        for j, p in enumerate(profile):
            diff = model.new_int_var(0, bound, f'absolute_gap_{s}_{j}')
            imbalance = numerator(j, p)
            model.add(diff >= imbalance)
            model.add(diff >= -imbalance)
            objective.append(weight * diff)
    d_weight = round(100 * policy.get('depth_weight', 0.20))
    e_weight = round(100 * policy.get('emerging_weight', 0.12))
    for j, p in enumerate(primary):
        job_depth = max(1.0, depth_target[j])
        # Credit-equivalent depth shortfall saturates when the target is reached.
        short = model.new_int_var(0, bound, f'depth_short_{j}')
        expr = sum((round(t * p) - round(b[j] / job_depth)) * v
                   for (v, _, _), m, b, t in zip(vectors, mass_coeff, depth_coeff, total_coeff))
        model.add(short >= expr)
        bonus = e_weight if kg['skills'][j]['stage'] in ('emerging', 'growing') else 0
        objective.append((d_weight + bonus) * short)
        if policy.get('jd_depth_weight', 0):
            exposure_short = model.new_int_var(0, bound, f'absolute_readiness_short_{j}')
            depth = sum(round(b[j] / job_depth) * v for (v, _, _), b in zip(vectors, depth_coeff))
            model.add(exposure_short >= 2*SCALE - depth)
            objective.append(round(1000*float(p)*policy['jd_depth_weight']) * exposure_short)
    for (c, v) in edits:
        objective.append((round(120*float(c.get('faculty_fte', 0.45))) + 20) * v)
    if policy.get('objective') == 'additive':
        # Same feasible region and exact same candidates as the distribution planner.
        mass_gain = sum(int(round(1000 * float(primary @ mass))) * v for v, mass, _ in vectors)
        depth_gain = sum(int(round(100 * float(primary @ depth))) * v for v, _, depth in vectors)
        edit_cost = sum(round(6*float(c.get('faculty_fte', 0.45))) * v for c, v in edits)
        model.minimize(edit_cost - mass_gain - depth_gain)
    else:
        model.minimize(sum(objective))
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_deterministic_time = 0.8 if joint else 0.3
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        core = solver.sufficient_assumptions_for_infeasibility() if status == cp_model.INFEASIBLE else []
        return {'status': solver.status_name(status), 'feasible': False, 'actions': [], 'iis_conflicts': [labels[i] for i in core], 'reconstructed_courses': base}

    def action(kind, c, credits, reason, delta_lab=0):
        item = {'action': kind, 'course_name': c['course_name'], 'credits': c['credits'],
                'delta_credits': credits, 'delta_lab_hours': delta_lab, 'semester': sem(c),
                'skills': list(_skills(c)), 'reason': reason}
        if c.get('module'):
            item['module'] = c['module']
        return item

    revised = {i: c for i, c, v in revisions if solver.value(v)}
    reconstructed, actions, selected_ids = [], [], []
    for i, (c, v) in enumerate(zip(base, keep)):
        if i in selection and solver.value(selection[i]):
            selected_ids.append(len(reconstructed))
        if not solver.value(v):
            actions.append(action('REMOVE_OR_MERGE_OUT', c, -c['credits'], '释放可替换选修学分', -float(c.get('lab_hours', 0))))
        elif i in revised:
            new = revised[i]
            new['upgraded_syllabus'] = True
            reconstructed.append(new)
            actions.append(action('UPGRADE_SYLLABUS', new, 0, '在原课程内以可考核模块替换部分课时', float(new.get('lab_hours', 0))-float(c.get('lab_hours', 0))))
        else:
            reconstructed.append(copy.deepcopy(c))
    for i, (c, v) in enumerate(additions, start=len(base)):
        if solver.value(v):
            if i in selection and solver.value(selection[i]):
                selected_ids.append(len(reconstructed))
            reconstructed.append(c)
            actions.append(action('ADD_COURSE', c, c['credits'], '净减少技能分布或深度缺口', float(c.get('lab_hours', 0))))
    a_cr, r_cr = sum(max(0, a['delta_credits']) for a in actions), -sum(min(0, a['delta_credits']) for a in actions)
    fac, labs = solver.value(fte)/100, solver.value(lab)/100
    checks = {
        'credit_conservation_ok': abs(sum(c['credits'] for c in base)-sum(c['credits'] for c in reconstructed)) <= slack/2+1e-8,
        'core_foundation_100pct_retained': all(solver.value(v) for c, v in zip(base, keep) if locked(c)),
        'faculty_fte_within_cap': fac <= fte_cap, 'lab_gpu_within_cap': labs <= lab_cap,
        'prereq_dag_valid': all(not solver.value(v) or any(solver.value(k) for k in ks) for v, ks in prerequisite_checks),
        'duplicate_free': all(solver.value(v) + solver.value(k) <= 1 for v, k in conflicts),
        'reform_budget_ok': max(a_cr, r_cr) <= changed_cap and len(revised) <= upgrades_cap,
        'semester_load_ok': all(sum(half(c) for c in reconstructed if sem(c) == s) <= max(56, sum(half(c) for c in base if sem(c) == s)) for s in range(1, 9)),
    }
    module_titles = [c['module']['title'] for c in revised.values() if 'title' in c['module']]
    checks['no_repeated_modules'] = len(module_titles) == len(set(module_titles))
    if strict and all('take_rate' in c for c in base):
        def expected_elective_credits(rows):
            return sum(c['credits'] * c['take_rate'] for c in rows if c.get('category') == 'major_elective' and not c.get('required'))
        checks['expected_elective_credits_retained'] = abs(expected_elective_credits(base) - expected_elective_credits(reconstructed)) < 1e-8
    prior_mass = sum((course_mass(c, index)[0] for c in base), np.zeros(len(index)))
    final_mass = sum((course_mass(c, index)[0] for c in reconstructed), np.zeros(len(index)))
    checks['core_skill_exposure_retained'] = all(final_mass[j] + 0.02 >= 0.80 * prior_mass[j]
                                                 for j, skill in enumerate(kg['skills']) if skill.get('core'))
    if joint:
        checks['path_credit_quota_ok'] = sum(half(reconstructed[i]) for i in selected_ids) == round(2*quota)
        checks['path_prerequisites_ok'] = all(not solver.value(v) or any(solver.value(k) for k in providers)
                                              for v, providers in selected_requirements)
        from solver.elective_path import realize_path
        delivery = realize_path(reconstructed, selected_ids, uptake)
        delivered = sum((course_mass(c, index)[0] for c in delivery), np.zeros(len(index)))
        checks['path_core_exposure_retained'] = all(delivered[j] + 0.02 >= 0.8*prior_mass[j]
                                                   for j, skill in enumerate(kg['skills']) if skill.get('core'))
    checks['zero_hard_violations'] = all(checks.values())
    return {'path_selected': selected_ids, 'status': solver.status_name(status), 'feasible': checks['zero_hard_violations'],
            'budget_mode': budget_mode, 'objective_value': solver.objective_value, 'objective_bound': solver.best_objective_bound,
            'solver_wall_time_ms': round(solver.wall_time*1000, 2),
            'before_tech_credits': sum(c['credits'] for c in base if c.get('category') != 'general_edu'),
            'after_tech_credits': sum(c['credits'] for c in reconstructed if c.get('category') != 'general_edu'),
            'removed_credits': r_cr, 'added_credits': a_cr, 'upgraded_core_count': len(revised),
            'faculty_fte_used': fac, 'faculty_fte_cap': fte_cap, 'lab_units_used': labs, 'lab_units_cap': lab_cap,
            'constraint_checks': checks, 'iis_conflicts': [], 'actions': actions, 'reconstructed_courses': reconstructed}
