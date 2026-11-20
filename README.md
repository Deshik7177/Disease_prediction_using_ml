# MediScope — Disease Prediction Using Machine Learning

A Flask symptom-assessment app using DDXPlus synthetic patient data and multinomial logistic regression.

## Important

The DDXPlus benchmark contains 49 pathologies and 223 evidence variables. It is synthetic and licensed for research/education; it is not clinically validated and must not be used for diagnosis or medical treatment.

## Setup

```powershell
cd "C:\Users\Deshik\OneDrive\Desktop\deshik projects\btech projests\Disease_prediction_using_ml"
python -m pip install -r requirements.txt
python train_model.py
waitress-serve --listen=127.0.0.1:5000 app:app
```

Open `http://127.0.0.1:5000` in a browser.

## Included

- Symptom selection interface
- Logistic regression prediction model
- Confidence score and alternative matches
- Medical-use disclaimer
- Responsive CSS
- Patient-facing symptom taxonomy
- Three-state PRESENT / ABSENT / UNKNOWN evidence handling
- Dependency-aware progressive follow-up questions
- Categorical and multi-choice DDXPlus normalization
- Urgency warning overrides
- Legacy `/api/predict` compatibility and `/api/followups`

## DDXPlus workflow

The primary selector is sourced from `data/symptom_taxonomy.json`. Entries map a patient-friendly label to an actual DDXPlus evidence ID. The raw `release_evidences.json` catalog remains the source of truth for evidence metadata.

- Binary evidence: PRESENT, ABSENT, or UNKNOWN
- Categorical evidence: an original `V_*` or numeric value
- Multi-choice evidence: one or more original `V_*` values
- `code_question` dependencies prevent unavailable follow-ups
- Unknown and absent evidence do not become positive model features

The API continues to accept legacy `symptoms: ["E_91"]` arrays. It also accepts:

```json
{
  "symptoms": ["E_91"],
  "evidence_states": {
    "E_91": {"state": "PRESENT", "value": null}
  },
  "age": 35,
  "sex": "M"
}
```

Run tests with:

```powershell
python -m pytest -q
```


#Tested by DR.yinti Revathi
