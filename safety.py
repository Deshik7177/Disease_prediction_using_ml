URGENT_SYMPTOMS = {
    "severe_breathing": ("E_66", "Severe breathing difficulty"),
    "chest_pain": ("E_14", "Chest pain"),
    "blood_cough": ("E_45", "Coughing up blood"),
    "loss_of_consciousness": ("E_128", "Loss of consciousness or severe choking"),
    "severe_bleeding": ("E_178", "Severe or unusual bleeding"),
}


def evaluate_safety(selected_evidence, evidence_states=None):
    selected = set(selected_evidence or [])
    states = evidence_states or {}
    for evidence_id, record in states.items():
        if isinstance(record, dict) and record.get("state") == "PRESENT":
            selected.add(evidence_id)

    emergency = [
        label
        for key, (evidence_id, label) in URGENT_SYMPTOMS.items()
        if evidence_id in selected
    ]
    if emergency:
        return {
            "urgent": True,
            "signals": emergency,
            "guidance": (
                "Urgent warning signs are present. Call local emergency services now "
                "or go to the nearest emergency department. Do not wait for a prediction."
            ),
        }
    return {"urgent": False, "signals": [], "guidance": None}
