import ast
import json
from pathlib import Path

import joblib
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, top_k_accuracy_score
from evidence_core import EvidenceNormalizer, parse_evidence_list


BASE_DIR = Path(__file__).resolve().parent
DDX_DIR = BASE_DIR / "data" / "ddxplus"
MODEL_PATH = BASE_DIR / "model" / "disease_model.joblib"


def evidence_matrix(frame, evidence_ids):
    evidence_lookup = {evidence_id: index for index, evidence_id in enumerate(evidence_ids)}
    rows = []
    columns = []
    values = []

    for row_index, evidence_value in enumerate(frame["EVIDENCES"]):
        evidence_items = parse_evidence_list(evidence_value)
        for evidence_id in evidence_items:
            column_index = evidence_lookup.get(evidence_id)
            if column_index is not None:
                rows.append(row_index)
                columns.append(column_index)
                values.append(1.0)

    evidence = csr_matrix(
        (values, (rows, columns)),
        shape=(len(frame), len(evidence_ids)),
        dtype="float32",
    )
    age = frame["AGE"].fillna(frame["AGE"].median()).astype("float32").to_numpy().reshape(-1, 1)
    age = age / 100.0
    sex = frame["SEX"].fillna("U").astype(str).map({"M": 1.0, "F": -1.0}).fillna(0.0)
    context = csr_matrix(sex.astype("float32").to_numpy().reshape(-1, 1))
    return hstack([evidence, csr_matrix(age), context], format="csr")


def load_split(path):
    frame = pd.read_csv(path)
    if not {"EVIDENCES", "PATHOLOGY", "AGE", "SEX"}.issubset(frame.columns):
        raise ValueError(f"{path.name} has an unexpected DDXPlus schema.")
    return frame


def train():
    evidence_catalog = json.loads((DDX_DIR / "release_evidences.json").read_text(encoding="utf-8"))
    normalizer = EvidenceNormalizer(DDX_DIR / "release_evidences.json")

    validation = load_split(DDX_DIR / "validate.csv")
    test = load_split(DDX_DIR / "test.csv")
    train_path = DDX_DIR / "train.csv"

    if train_path.exists():
        training = load_split(train_path)
        print(f"Using official DDXPlus training split: {len(training):,} patients")
    else:
        training = validation
        print("train.csv is not downloaded; using validate.csv as a 132k-patient training fallback")

    feature_ids = set(normalizer.valid_tokens)
    for frame in (training, validation, test):
        for raw_tokens in frame["EVIDENCES"]:
            feature_ids.update(
                token
                for token in parse_evidence_list(raw_tokens)
                if token in normalizer.valid_tokens
            )
    feature_ids = sorted(feature_ids)

    x_train = evidence_matrix(training, feature_ids)
    y_train = training["PATHOLOGY"].astype(str)

    model = LogisticRegression(
        solver="saga",
        max_iter=120,
        C=1.0,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(x_train, y_train)

    x_test = evidence_matrix(test, feature_ids)
    y_test = test["PATHOLOGY"].astype(str)
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)
    accuracy = accuracy_score(y_test, predictions)
    balanced_accuracy = balanced_accuracy_score(y_test, predictions)
    top5 = top_k_accuracy_score(y_test, probabilities, k=5, labels=model.classes_)

    evidence_labels = {
        evidence_id: details.get("question_en", evidence_id)
        for evidence_id, details in evidence_catalog.items()
    }
    condition_names = sorted(set(y_train).union(set(y_test)))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_ids": feature_ids,
            "evidence_ids": sorted(normalizer.valid_tokens),
            "evidence_labels": evidence_labels,
            "conditions": condition_names,
            "accuracy": float(accuracy),
            "balanced_accuracy": float(balanced_accuracy),
            "top5_accuracy": float(top5),
            "dataset": "DDXPlus",
        },
        MODEL_PATH,
    )

    print(f"Model saved: {MODEL_PATH}")
    print(f"Test accuracy: {accuracy:.1%}")
    print(f"Balanced accuracy: {balanced_accuracy:.1%}")
    print(f"Top-5 accuracy: {top5:.1%}")


if __name__ == "__main__":
    train()
