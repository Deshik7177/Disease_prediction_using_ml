from pathlib import Path

import joblib
from flask import Flask, jsonify, render_template, request
import json

from evidence_core import EvidenceNormalizer
from question_engine import AdaptiveQuestionEngine
from safety import evaluate_safety


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "disease_model.joblib"

app = Flask(__name__)

if not MODEL_PATH.exists():
    raise RuntimeError("Model not found. Run: python train_model.py")

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
evidence_ids = bundle["evidence_ids"]
feature_ids = bundle.get("feature_ids", evidence_ids)
evidence_labels = bundle["evidence_labels"]
accuracy = bundle["accuracy"]
balanced_accuracy = bundle["balanced_accuracy"]
top5_accuracy = bundle["top5_accuracy"]

evidence_lookup = set(feature_ids)
normalizer = EvidenceNormalizer()
with (BASE_DIR / "data" / "symptom_taxonomy.json").open(encoding="utf-8") as taxonomy_file:
    taxonomy = json.load(taxonomy_file)
taxonomy_lookup = {item["evidence_id"]: item for item in taxonomy}
question_engine = AdaptiveQuestionEngine(
    normalizer,
    normalizer.catalog,
    model,
    feature_ids,
)


def evidence_matrix(selected, age, sex, evidence_states=None):
    if len(selected) > 40:
        raise ValueError("A maximum of 40 evidence items is allowed.")
    return normalizer.encode_sparse(
        normalizer.normalize_to_positive_tokens(
            evidence_states or {},
            legacy_positive_ids=selected,
        ),
        feature_ids,
        age,
        sex,
    )


def predict(selected, age, sex, evidence_states=None):
    if not selected:
        return {"error": "Select at least one symptom or sign."}, 400

    features = evidence_matrix(selected, age, sex, evidence_states)
    probabilities = model.predict_proba(features)[0]
    ranked = sorted(
        zip(model.classes_, probabilities),
        key=lambda item: item[1],
        reverse=True,
    )
    top_probability = ranked[0][1]
    if len(selected) < 3:
        uncertainty = "Insufficient evidence"
    elif top_probability < 0.20:
        uncertainty = "Low confidence"
    elif top_probability < 0.45:
        uncertainty = "Moderate confidence"
    else:
        uncertainty = "Higher confidence"
    safety = evaluate_safety(selected, evidence_states)
    supporting_evidence = [
        taxonomy_lookup[item]["display_name"]
        for item in selected
        if item in taxonomy_lookup
    ]
    if len(selected) < 3:
        missing_information = "Location, onset, severity and duration have not yet been collected."
    else:
        missing_information = "Additional follow-up details may improve the differential."
    next_step = (
        "Use the ranked list only as decision support and consult a qualified clinician."
    )

    return {
        "prediction": ranked[0][0],
        "confidence": round(top_probability * 100, 1),
        "uncertainty": uncertainty,
        "calibration": "Not clinically calibrated",
        "alternatives": [
            {"condition": condition, "probability": round(probability * 100, 1)}
            for condition, probability in ranked[1:6]
        ],
        "supporting_evidence": supporting_evidence,
        "missing_information": missing_information,
        "next_step": next_step,
        "safety": safety,
        "metrics": {
            "test_accuracy": round(accuracy * 100, 1),
            "balanced_accuracy": round(balanced_accuracy * 100, 1),
            "top5_accuracy": round(top5_accuracy * 100, 1),
        },
    }, 200


@app.get("/")
def index():
    return render_template(
        "index.html",
        taxonomy=taxonomy,
        accuracy=accuracy,
        balanced_accuracy=balanced_accuracy,
        top5_accuracy=top5_accuracy,
    )


@app.post("/api/predict")
def api_predict():
    payload = request.get_json(silent=True) or {}
    selected = payload.get("symptoms", [])
    evidence_states = payload.get("evidence_states", {})
    if not isinstance(selected, list):
        return jsonify(error="Symptoms must be a list."), 400
    unknown = [item for item in selected if item not in evidence_lookup]
    if unknown:
        return jsonify(error="One or more evidence items are invalid."), 400
    if evidence_states and not isinstance(evidence_states, dict):
        return jsonify(error="evidence_states must be an object."), 400
    try:
        age = int(payload.get("age", 30))
    except (TypeError, ValueError):
        return jsonify(error="Age must be a number."), 400
    sex = str(payload.get("sex", "U")).upper()
    try:
        states = normalizer.parse_patient_states(evidence_states)
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    unknown_states = [item for item in states if item not in evidence_lookup]
    if unknown_states:
        return jsonify(error="One or more evidence state IDs are invalid."), 400
    result, status = predict(selected, age, sex, states)
    return jsonify(result), status


@app.post("/api/followups")
def api_followups():
    payload = request.get_json(silent=True) or {}
    evidence_states = payload.get("evidence_states", {})
    if not isinstance(evidence_states, dict):
        return jsonify(error="evidence_states must be an object."), 400
    try:
        states = normalizer.parse_patient_states(evidence_states)
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    unknown = [item for item in states if item not in evidence_lookup]
    if unknown:
        return jsonify(error="One or more evidence state IDs are invalid."), 400
    try:
        age = int(payload.get("age", 30))
        limit = int(payload.get("limit", 6))
    except (TypeError, ValueError):
        return jsonify(error="Age and limit must be integers."), 400
    limit = max(1, min(limit, 20))
    sex = str(payload.get("sex", "U")).upper()
    selected = [
        evidence_id
        for evidence_id, record in states.items()
        if record["state"] == "PRESENT"
    ]
    safety = evaluate_safety(selected, states)
    if safety["urgent"]:
        return jsonify(
            safety=safety,
            questions=[],
            stop=True,
        )
    questions = question_engine.next_questions(
        states,
        taxonomy,
        age,
        sex,
        limit,
    )
    return jsonify(safety=safety, questions=questions, stop=False)


@app.post("/predict")
def web_predict():
    selected = request.form.getlist("symptoms")
    try:
        age = int(request.form.get("age", "30"))
    except ValueError:
        age = 30
    sex = request.form.get("sex", "U")
    raw_states = request.form.get("evidence_states", "{}")
    try:
        states = normalizer.parse_patient_states(json.loads(raw_states))
    except (json.JSONDecodeError, ValueError):
        states = {}
    for evidence_id in selected:
        states.setdefault(evidence_id, {"state": "PRESENT", "value": None})
    result, status = predict(selected, age, sex, states)
    if status != 200:
        return render_template(
            "index.html",
            taxonomy=taxonomy,
            accuracy=accuracy,
            balanced_accuracy=balanced_accuracy,
            top5_accuracy=top5_accuracy,
            error=result["error"],
        )
    return render_template(
        "index.html",
        taxonomy=taxonomy,
        accuracy=accuracy,
        balanced_accuracy=balanced_accuracy,
        top5_accuracy=top5_accuracy,
        result=result,
        selected=selected,
        age=age,
        sex=sex,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
