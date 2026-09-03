import random

from build_cpu_yolo_protocol import DOMAINS, allocate, sample_domain


def rows(positive_count=70, negative_count=30):
    positives = [{"id": f"p{i}", "negatives": "0"} for i in range(positive_count)]
    negatives = [{"id": f"n{i}", "negatives": "1"} for i in range(negative_count)]
    return positives + negatives


def test_allocate_is_exact_and_balanced():
    quotas = allocate(2800, list(DOMAINS))
    assert sum(quotas.values()) == 2800
    assert set(quotas.values()) == {400}


def test_sampling_preserves_rounded_negative_prevalence():
    selected = sample_domain(rows(), 40, random.Random(2026))
    assert len(selected) == 40
    assert sum(int(row["negatives"]) for row in selected) == 12
    assert len({row["id"] for row in selected}) == 40


def test_sampling_is_seed_deterministic():
    first = sample_domain(rows(), 40, random.Random("2026:joint"))
    second = sample_domain(rows(), 40, random.Random("2026:joint"))
    assert [row["id"] for row in first] == [row["id"] for row in second]
