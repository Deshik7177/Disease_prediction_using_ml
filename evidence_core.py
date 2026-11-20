import ast
import json
from pathlib import Path

from scipy.sparse import csr_matrix, hstack


BASE_DIR = Path(__file__).resolve().parent
EVIDENCE_CATALOG_PATH = BASE_DIR / "data" / "ddxplus" / "release_evidences.json"


class EvidenceNormalizer:
    def __init__(self, catalog_path=EVIDENCE_CATALOG_PATH):
        self.catalog = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
        self.base_ids = set(self.catalog)
        self.valid_tokens = set()
        for evidence_id, metadata in self.catalog.items():
            self.valid_tokens.add(evidence_id)
            values = metadata.get("possible-values") or []
            for value in values:
                self.valid_tokens.add(f"{evidence_id}_@_{value}")

    def is_valid(self, evidence_id):
        return evidence_id in self.valid_tokens

    def base_id(self, token):
        return str(token).split("_@_", 1)[0]

    def parse_patient_states(self, states):
        normalized = {}
        if isinstance(states, dict):
            for evidence_id, value in states.items():
                if isinstance(value, dict):
                    state = str(value.get("state", "UNKNOWN")).upper()
                    selected_value = value.get("value")
                else:
                    state = str(value).upper()
                    selected_value = None

                if state not in {"PRESENT", "ABSENT", "UNKNOWN"}:
                    raise ValueError("Evidence state must be PRESENT, ABSENT, or UNKNOWN.")
                normalized[evidence_id] = {
                    "state": state,
                    "value": selected_value,
                }
        return normalized

    def normalize_to_positive_tokens(self, states, legacy_positive_ids=None):
        states = self.parse_patient_states(states)
        positive_tokens = set()

        for evidence_id in legacy_positive_ids or []:
            if evidence_id in self.valid_tokens:
                positive_tokens.add(evidence_id)

        for evidence_id, record in states.items():
            if record["state"] != "PRESENT":
                continue
            if record["value"] is not None:
                values = record["value"] if isinstance(record["value"], list) else [record["value"]]
                for value in values:
                    token = f"{evidence_id}_@_{value}"
                    if token in self.valid_tokens:
                        positive_tokens.add(token)
            elif evidence_id in self.valid_tokens:
                positive_tokens.add(evidence_id)

        return positive_tokens

    def encode_sparse(self, positive_tokens, feature_ids, age=30, sex="U"):
        feature_lookup = {feature_id: index for index, feature_id in enumerate(feature_ids)}
        columns = [
            feature_lookup[token]
            for token in positive_tokens
            if token in feature_lookup
        ]
        evidence = csr_matrix(
            ([1.0] * len(columns), ([0] * len(columns), columns)),
            shape=(1, len(feature_ids)),
            dtype="float32",
        )
        normalized_age = csr_matrix([[max(0, min(int(age), 120)) / 100.0]])
        sex_value = {"M": 1.0, "F": -1.0}.get(str(sex).upper(), 0.0)
        context = csr_matrix([[sex_value]], dtype="float32")
        return hstack([evidence, normalized_age, context], format="csr")

    def evidence_dependencies(self, evidence_id):
        metadata = self.catalog.get(evidence_id)
        if not metadata:
            return None
        parent = metadata.get("code_question")
        if parent == evidence_id:
            return None
        return parent

    def is_available(self, evidence_id, answered_states):
        dependency = self.evidence_dependencies(evidence_id)
        if dependency is None:
            return True
        record = answered_states.get(dependency)
        return bool(record and record.get("state") == "PRESENT")


def parse_evidence_list(raw):
    try:
        parsed = ast.literal_eval(str(raw))
        return parsed if isinstance(parsed, list) else []
    except (SyntaxError, ValueError):
        return []
