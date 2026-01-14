import datetime
import random
from collections import defaultdict
from datetime import timedelta

from .facts import (
    ChineseZodiacFact,
    DayOfMonthFact,
    DayOfMonthRangeFact,
    DecadeFact,
    EventFact,
    ExplicitYearFact,
    FactLevel,
    KnowledgeBaseEventFact,
    LeapYearFact,
    LunarMonthFact,
    MonthFact,
    MultiWeekdayFact,
    PersonAliveFact,
    SeasonFact,
    USPresidentFact,
    WeekdayFact,
    WeekdayInMonthFact,
)

# The fact we are not using currently
# DayOfYearFact: too specific
# WeekdayInYearFact: too specific
# WesternZodiacFact: varying over years


class ConstraintGenerator:
    def __init__(self, start_year=1950, end_year=2050):
        self.start_year = start_year
        self.end_year = end_year

        # Initialize fact generators
        self.fact_generators = [
            WeekdayFact(),
            MultiWeekdayFact(),
            DayOfMonthRangeFact(),
            MonthFact(),
            LeapYearFact(),
            SeasonFact(),
            WeekdayInMonthFact(),
            DayOfMonthFact(),
            ExplicitYearFact(),
            DecadeFact(),
            ChineseZodiacFact(),
            LunarMonthFact(),
            PersonAliveFact(),
            USPresidentFact(),
            EventFact(),
        ]

        self.kb_event_fact_generator = KnowledgeBaseEventFact(
            year_range=(self.start_year, self.end_year)
        )

    def _get_random_date(self, start_year, end_year):
        start_date = datetime.date(start_year, 1, 1)
        end_date = datetime.date(end_year, 12, 31)

        delta = end_date - start_date
        random_days = random.randint(0, delta.days)
        return start_date + timedelta(days=random_days)

    def generate_puzzle(self, num_constraints, allowed_solution_counts):
        if num_constraints < 3:
            raise ValueError("num_constraints must be at least 3")

        while True:
            # 1. Pick a seed date
            kb_fact = None
            # We use KB facts even if include_common_knowledge is False,
            # but they will be formatted explicitly (as date constraints).
            if True:
                seed_date, kb_meta = (
                    self.kb_event_fact_generator.get_random_fact_and_date()
                )
                kb_fact = {
                    "meta": kb_meta,
                    "generator": self.kb_event_fact_generator,
                    "description": self.kb_event_fact_generator.format(
                        kb_meta, explicit=False
                    ),
                    "explicit_description": self.kb_event_fact_generator.format(
                        kb_meta, explicit=True
                    ),
                    "is_noise": False,
                }
            else:
                seed_date = self._get_random_date(self.start_year, self.end_year)

            # Generate Fact Schedule
            # We want the final distribution to be balanced.
            levels = [FactLevel.YEAR, FactLevel.MONTH, FactLevel.DAY]
            base_count = num_constraints // 3
            remainder = num_constraints % 3

            schedule = {level: base_count for level in levels}
            random.shuffle(levels)
            for i in range(remainder):
                schedule[levels[i]] += 1

            if kb_fact:
                kb_level = self.kb_event_fact_generator.level(kb_fact["meta"])
                if schedule[kb_level] > 0:
                    schedule[kb_level] -= 1
                else:
                    # If the slot for this level is already full (0), we need to remove
                    # a slot from another level to maintain the total count.
                    candidates = [l for l in FactLevel if schedule[l] > 0]
                    if candidates:
                        chosen = random.choice(candidates)
                        schedule[chosen] -= 1

            # Handle zero-solution case
            # Enable noise if 0 is in allowed_solution_counts
            use_noise = 0 in allowed_solution_counts
            noise_date = None
            if use_noise:
                # Generate noise date within the global range
                noise_date = self._get_random_date(self.start_year, self.end_year)

            # Generate all candidates
            candidates_by_level = {level: defaultdict(list) for level in FactLevel}

            generators = self.fact_generators.copy()
            random.shuffle(generators)

            for generator in generators:
                # Generate facts for seed_date
                metas = generator.generate(seed_date)
                for m in metas:
                    lvl = generator.level(m)
                    candidates_by_level[lvl][generator].append(
                        {
                            "meta": m,
                            "generator": generator,
                            "description": generator.format(m, explicit=False),
                            "explicit_description": generator.format(m, explicit=True),
                            "is_noise": False,
                        }
                    )

                if use_noise:
                    noise_metas = generator.generate(noise_date)
                    for m in noise_metas:
                        lvl = generator.level(m)
                        candidates_by_level[lvl][generator].append(
                            {
                                "meta": m,
                                "generator": generator,
                                "description": generator.format(m, explicit=False),
                                "explicit_description": generator.format(
                                    m, explicit=True
                                ),
                                "is_noise": True,
                            }
                        )

            selected_facts = []
            has_weekday_fact = False
            used_generators = set()

            if kb_fact:
                selected_facts.append(kb_fact)
                used_generators.add(self.kb_event_fact_generator)

            # Try to fulfill the schedule
            target_levels = list(FactLevel)
            random.shuffle(target_levels)

            for level in target_levels:
                count_needed = schedule[level]
                if count_needed == 0:
                    continue

                candidates_by_gen = candidates_by_level[level]
                available_gens = list(candidates_by_gen.keys())
                random.shuffle(available_gens)

                facts_added_for_level = 0
                for gen in available_gens:
                    if facts_added_for_level >= count_needed:
                        break

                    # when noise, some generators may have more than one candidate
                    # we only want to pick one per generator
                    if gen in used_generators:
                        continue

                    if gen.is_weekday_related and has_weekday_fact:
                        continue

                    # Pick a random candidate from this generator
                    gen_candidates = candidates_by_gen[gen]
                    selected = random.choice(gen_candidates)

                    selected_facts.append(selected)
                    used_generators.add(gen)
                    facts_added_for_level += 1

                    if gen.is_weekday_related:
                        has_weekday_fact = True

                assert facts_added_for_level >= count_needed, (
                    f"Error generating date {seed_date}. "
                    f"Not enough for level {level}: "
                    f"needed {count_needed}, "
                    f"found {facts_added_for_level}."
                )

            assert len(selected_facts) == num_constraints, (
                f"Error generating date {seed_date}. "
                f"Expected {num_constraints} facts, "
                f"but selected {len(selected_facts)}"
            )

            # 4. Solve
            solutions = self._solve(selected_facts)

            # 5. Check quality
            sol_count = len(solutions)
            if sol_count not in allowed_solution_counts:
                continue

            # Shuffle constraints for display
            display_constraints = selected_facts.copy()
            random.shuffle(display_constraints)

            return {
                "seed_date": str(seed_date),
                "noise_date": str(noise_date) if noise_date else None,
                "constraints": [c["description"] for c in display_constraints],
                "explicit_constraints": [
                    c["explicit_description"] for c in display_constraints
                ],
                "constraints_meta": [c["meta"] for c in display_constraints],
                "solutions": solutions,
                "solution_count": len(solutions),
                "year_range": [self.start_year, self.end_year],
            }

    def _generate_facts(self, date):
        facts = []
        for generator in self.fact_generators:
            metas = generator.generate(date)
            for meta in metas:
                facts.append(
                    {
                        "meta": meta,
                        "generator": generator,
                        "description": generator.format(meta),
                    }
                )
        return facts

    def _solve(self, constraints):
        start_date = datetime.date(self.start_year, 1, 1)
        end_date = datetime.date(self.end_year, 12, 31)

        # 1. Generate initial list of dates
        candidate_dates = []
        curr = start_date
        while curr <= end_date:
            candidate_dates.append(curr)
            curr += timedelta(days=1)

        # 2. Sort constraints by information gain
        constraints.sort(
            key=lambda c: c["generator"].get_information_gain(c["meta"]), reverse=True
        )

        # 3. Filter dates
        for c in constraints:
            if not candidate_dates:
                break

            generator = c["generator"]
            meta = c["meta"]

            # Filter
            candidate_dates = [
                d for d in candidate_dates if generator.validate(d, meta)
            ]

        # 4. Convert to strings
        return [str(d) for d in candidate_dates]
