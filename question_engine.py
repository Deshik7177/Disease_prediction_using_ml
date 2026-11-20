import math


class AdaptiveQuestionEngine:
    def __init__(self, normalizer, catalog, model, feature_ids, default_limit=6):
        self.normalizer = normalizer
        self.catalog = catalog
        self.model = model
        self.feature_ids = feature_ids
        self.default_limit = default_limit

    def _encode(self, states, age, sex):
        positive = self.normalizer.normalize_to_positive_tokens(states)
        from evidence_core import EvidenceNormalizer
        return self.normalizer.encode_sparse(positive, self.feature_ids, age, sex)

    def candidate_questions(self, states, taxonomy, limit=None):
        limit = limit or self.default_limit
        selected_ids = {
            evidence_id
            for evidence_id, record in states.items()
            if record.get("state") == "PRESENT"
        }
        related = set(selected_ids)
        for item in taxonomy:
            if item["evidence_id"] in selected_ids:
                related.update(item.get("followup_evidence_ids", []))

        candidates = []
        for evidence_id in related:
            if evidence_id not in self.catalog:
                continue
            metadata = self.catalog[evidence_id]
            if metadata.get("is_antecedent"):
                continue
            if evidence_id in states:
                continue
            if not self.normalizer.is_available(evidence_id, states):
                continue
            candidates.append({
                "evidence_id": evidence_id,
                "question": metadata.get("question_en") or metadata.get("name") or evidence_id,
                "type": metadata.get("data_type"),
                "values": metadata.get("possible-values") or [],
                "value_meaning": metadata.get("value_meaning") or {},
            })

        return candidates[:limit]

    def next_questions(self, states, taxonomy, age=30, sex="U", limit=None):
        candidates = self.candidate_questions(states, taxonomy, limit=50)
        ranked = candidates
        try:
            features = self.normalizer.encode_sparse(
                self.normalizer.normalize_to_positive_tokens(states),
                self.feature_ids,
                age,
                sex,
            )
            probabilities = self.model.predict_proba(features)[0]
            entropy = -sum(
                probability * math.log(probability + 1e-12)
                for probability in probabilities
            )
            ranked = sorted(
                candidates,
                key=lambda candidate: (
                    candidate["type"] == "B",
                    -entropy,
                ),
            )
        except (ValueError, AttributeError):
            ranked = candidates
        return ranked[: (limit or self.default_limit)]
