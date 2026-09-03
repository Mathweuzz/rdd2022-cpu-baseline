import pandas as pd

from prepare_rdd2022 import SPLITS, assign_domain_groups, sequence_number


def test_sequence_number_uses_final_numeric_component():
    assert sequence_number("Japan_001234.jpg") == 1234
    assert sequence_number("road_12_frame_98.png") == 98


def test_group_assignment_is_deterministic_and_exclusive():
    groups = pd.DataFrame(
        {
            "group_id": [f"Demo:{index}" for index in range(15)],
            "domain": ["Demo"] * 15,
            "images": [50] * 15,
            "negatives": [index % 9 for index in range(15)],
            "D00": [20 + index for index in range(15)],
            "D10": [10 + index % 4 for index in range(15)],
            "D20": [8 + index % 3 for index in range(15)],
            "D40": [3 + index % 2 for index in range(15)],
        }
    )
    first = assign_domain_groups(groups, seed=2026)
    second = assign_domain_groups(groups, seed=2026)
    assert first == second
    assert set(first) == set(groups["group_id"])
    assert set(first.values()).issubset(set(SPLITS))
    assert len(first) == len(groups)
