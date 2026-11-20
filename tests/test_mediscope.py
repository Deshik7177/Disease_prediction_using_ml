import json
from pathlib import Path

import pytest

from evidence_core import EvidenceNormalizer
from safety import evaluate_safety


BASE_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def normalizer():
    return EvidenceNormalizer()


def test_catalog_has_expected_sizes(normalizer):
    assert len(normalizer.catalog) == 223
    assert sum(item.get("is_antecedent", False) for item in normalizer.catalog.values()) == 113


def test_taxonomy_entries_map_to_real_evidence(normalizer):
    taxonomy = json.loads((BASE_DIR / "data" / "symptom_taxonomy.json").read_text(encoding="utf-8"))
    categories = {
        "General symptoms",
        "Respiratory",
        "Cardiovascular",
        "Neurological",
        "Digestive",
        "Skin and dermatological",
        "Musculoskeletal",
        "Eyes and ENT",
        "Urinary",
        "Reproductive",
    }
    assert {item["category"] for item in taxonomy} <= categories
    for item in taxonomy:
        assert normalizer.is_valid(item["evidence_id"])
        assert item["evidence_type"] in {"binary", "categorical", "multi-choice"}
        assert item["clinical_review_status"] == "ddxplus_verified"


def test_unknown_does_not_add_positive_evidence(normalizer):
    states = {
        "E_91": {"state": "UNKNOWN", "value": None},
        "E_201": {"state": "ABSENT", "value": None},
    }
    assert normalizer.normalize_to_positive_tokens(states) == set()


def test_present_binary_adds_positive_evidence(normalizer):
    states = {"E_91": {"state": "PRESENT", "value": None}}
    assert normalizer.normalize_to_positive_tokens(states) == {"E_91"}


def test_categorical_value_retains_valid_qualified_token(normalizer):
    states = {
        "E_130": {"state": "PRESENT", "value": "V_107"},
    }
    assert normalizer.normalize_to_positive_tokens(states) == {"E_130_@_V_107"}


def test_invalid_categorical_value_is_rejected(normalizer):
    states = {
        "E_130": {"state": "PRESENT", "value": "invalid"},
    }
    assert normalizer.normalize_to_positive_tokens(states) == set()


def test_conditional_dependency_is_respected(normalizer):
    states = {"E_53": {"state": "ABSENT", "value": None}}
    assert not normalizer.is_available("E_56", states)
    states["E_53"] = {"state": "PRESENT", "value": None}
    assert normalizer.is_available("E_56", states)


def test_invalid_evidence_state_is_rejected(normalizer):
    with pytest.raises(ValueError):
        normalizer.parse_patient_states({"E_91": {"state": "MAYBE", "value": None}})


def test_urgency_override_stops_for_chest_pain():
    result = evaluate_safety(["E_14"], {"E_14": {"state": "PRESENT"}})
    assert result["urgent"]
    assert "chest pain" in result["guidance"].lower() or "Chest pain" in result["signals"]


def test_urgent_absent_evidence_is_not_urgent():
    result = evaluate_safety(
        [],
        {
            "E_14": {"state": "ABSENT"},
            "E_66": {"state": "UNKNOWN"},
        },
    )
    assert not result["urgent"]


def test_multi_choice_values_retain_all_valid_tokens(normalizer):
    states = {
        "E_55": {"state": "PRESENT", "value": ["V_29", "V_101"]},
    }
    assert normalizer.normalize_to_positive_tokens(states) == {
        "E_55_@_V_29",
        "E_55_@_V_101",
    }


def test_invalid_binary_id_is_rejected(normalizer):
    assert not normalizer.is_valid("not_a_ddxplus_id")
