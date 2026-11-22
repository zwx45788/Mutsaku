"""Transparent expected-credit scenarios and auditable catalogue comparability."""
from __future__ import annotations

import re
from collections import defaultdict


from analytics.source_curricula import reconcile


def expected_pathway(courses, credit_target=None):
    """Preserve listed required courses and scale only the eligible elective catalogue.

    The common credit target is an assumption for comparison, never a recovered graduation rule.
    A catalogue with too many required credits or too few total credits is explicitly flagged.
    """
    courses, source = reconcile(courses)
    use_official_quota = credit_target is None and source is not None
    if credit_target is None:
        credit_target = source['graduation_credits'] if source else 160.0
    required = sum(c['credits'] for c in courses if c.get('required'))
    optional = sum(c['credits'] for c in courses if not c.get('required'))
    optional_take = min(1.0, max(0.0, (credit_target - required)/optional)) if optional else 0.0
    tech_elective_rate = optional_take
    tech_options = sum(c['credits'] for c in courses if not c.get('required') and c.get('category') == 'major_elective')
    official_minimum = source['technical_elective_minimum'] if use_official_quota else None
    if official_minimum is not None and tech_options:
        tech_elective_rate = min(1.0, official_minimum / tech_options)
        remaining_options = optional - tech_options
        if remaining_options > 0:
            optional_take = min(1.0, max(0.0, (credit_target - required - min(official_minimum, tech_options)) / remaining_options))
    pathway = [dict(c, take_rate=1.0 if c.get('required') else
                    tech_elective_rate if c.get('category') == 'major_elective' else optional_take)
               for c in courses]
    credit_mass = [c['credits'] * c['take_rate'] for c in pathway]
    planned = sum(credit_mass)
    technical = sum(m for c, m in zip(pathway, credit_mass) if c.get('category') != 'general_edu')
    confirmed_semester = sum(m for c, m in zip(pathway, credit_mass) if 1 <= int(c.get('semester_raw', c.get('semester')) or 0) <= 8)
    official_syllabus = sum(m for c, m in zip(pathway, credit_mass) if c.get('has_syllabus') and c.get('category') != 'general_edu')
    return pathway, {
        'catalogue_credits': required + optional, 'required_catalogue_credits': required,
        'expected_credits': round(planned, 4), 'elective_take_rate': round(optional_take, 4),
        'credit_target': credit_target, 'official_requirements': source,
        'technical_elective_take_rate': round(tech_elective_rate, 4), 'syllabus_credit_share': round(official_syllabus / max(1, technical), 4),
        'source_semester_share': round(confirmed_semester / max(1, planned), 4),
        'status': ('required_catalogue_exceeds_budget' if required > credit_target else
                   'incomplete_catalogue' if required + optional < credit_target else 'assumption_compatible'),
    }


def role_family(title):
    title = title.lower()
    if re.search(r'算法|研究员|科学家|研究工程师|科研|芯片|编译器|内核|基础软件|模型训练|research scientist', title):
        return 'research'
    if re.search(r'实习|intern', title):
        return 'intern'
    return 'engineering'


def tier_default(tier):
    if tier.startswith('tier1'):
        return 'research'
    if tier.startswith('tier2'):
        return 'technical'
    return 'broad_entry'


def group_curricula(courses):
    groups = defaultdict(list)
    for course in courses:
        groups[course['program_id']].append(course)
    return groups
