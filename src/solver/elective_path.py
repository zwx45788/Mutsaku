"""Recommends an elective path while preserving the course catalogue and credit quota."""
from __future__ import annotations

import numpy as np
from ortools.sat.python import cp_model

from solver.distribution_solver import near_duplicate


def _credits(course):
    return int(round(2 * course['credits']))


def _weight(course):
    return float(course.get('take_rate', 1.0 if course.get('required') else 0.55))


def realize_path(courses, selected, uptake):
    if not 0 <= uptake <= 1:
        raise ValueError('Uptake must be between zero and one')
    selected = set(selected)
    return [dict(course, take_rate=(1-uptake)*_weight(course) + uptake*int(i in selected))
            if course.get('category') == 'major_elective' and not course.get('required') else dict(course)
            for i, course in enumerate(courses)]


def recommend_elective_path(courses, demand_dist, skill_ids, *, quota_credits, uptake=0.5,
                            objective='distribution', semester_cap=28):
    """Optimize the skill distribution of a recommendation at a prespecified uptake.

    Borrow no credits from the required block. All elective recommendations need a verified
    semester or a planned new offering. Existing students can retain the original choice set.
    """
    index = {skill: i for i, skill in enumerate(skill_ids)}
    rows = [(i, c) for i, c in enumerate(courses) if c.get('category') == 'major_elective' and not c.get('required')]
    model = cp_model.CpModel()
    take = {i: model.new_bool_var(f'course_{i}') for i, _ in rows}
    quota_units = round(2*quota_credits)
    model.add(sum(_credits(c)*take[i] for i, c in rows) == quota_units)
    for i, course in rows:
        term = int(course.get('semester') or 0)
        if term not in range(1, 9) or course.get('term_evidence') == 'unverified':
            model.add(take[i] == 0)
        for skill in course.get('prereq_skills', ()):
            providers = [(j, c) for j, c in enumerate(courses) if 0 < int(c.get('semester') or 0) < term
                         and any(s['id'] == skill and s['weight'] >= 0.10 for s in c.get('extracted_skills', []))]
            if not any(c.get('required') for j, c in providers):
                model.add(take[i] <= sum(take[j] for j, _ in providers if j in take))
    for offset, (i, c) in enumerate(rows):
        for j, other in rows[:offset]:
            if near_duplicate(c, other):
                model.add(take[i] + take[j] <= 1)
    for term in range(1, 9):
        required_term = sum(_credits(c) for c in courses if c.get('required') and c.get('semester') == term)
        model.add(sum(_credits(c)*take[i] for i, c in rows if c.get('semester') == term)
                  <= max(0, 2*semester_cap-required_term))

    base_mass = np.zeros(len(index))
    variable_mass = {}
    for i, course in enumerate(courses):
        if course.get('category') == 'general_edu':
            continue
        mass = np.zeros(len(index))
        for skill in course.get('extracted_skills', []):
            if skill['id'] in index:
                mass[index[skill['id']]] += float(course['credits']) * skill['weight']
        base_mass += mass * _weight(course) * (1-uptake if i in take else 1)
        if i in take:
            variable_mass[i] = mass * uptake
    if objective == 'additive':
        model.maximize(sum(round(100000*float(demand_dist @ mass))*take[i] for i, mass in variable_mass.items()))
    elif objective == 'distribution':
        gaps = []
        for j, target in enumerate(demand_dist):
            gap = model.new_int_var(0, 1_000_000, f'gap_{j}')
            imbalance = round(1000*(base_mass[j]-target*base_mass.sum()))
            imbalance += sum(round(1000*(mass[j]-target*mass.sum()))*take[i] for i, mass in variable_mass.items())
            model.add(gap >= imbalance)
            model.add(gap >= -imbalance)
            gaps.append(gap)
        model.minimize(sum(gaps))
    else:
        raise ValueError(f'Unknown elective objective: {objective}')
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_deterministic_time = 0.15
    state = solver.solve(model)
    if state not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {'status': solver.status_name(state), 'selected': [], 'quota_credits': quota_credits}
    selected = [i for i in take if solver.value(take[i])]
    return {'status': solver.status_name(state), 'selected': selected, 'quota_credits': quota_credits,
            'courses': [{'name': courses[i]['course_name'], 'credits': courses[i]['credits'],
                         'semester': courses[i]['semester']} for i in selected]}
