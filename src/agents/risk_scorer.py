import json
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from src.rag.retriever import OSHARetriever
from src.prediction.predictor import InjuryPredictor

load_dotenv()

# map each YOLO violation class to relevant OSHA event + source
VIOLATION_EVENT_MAP = {
    "Without Helmet": {
        "event": "Struck by falling object or equipment, n.e.c.",
        "source": "Floor, n.e.c.",
        "hazard": "Head injury from impact, falling/flying objects, electrical shock",
    },
    "Without Vest": {
        "event": "Struck by powered vehicle, unspecified",
        "source": "Forklift, order picker, platform truck-powered",
        "hazard": "Low visibility, struck-by hazard from vehicles/equipment",
    },
    "Without Glove": {
        "event": "Caught in running equipment or machinery during regular operation",
        "source": "Machinery, unspecified",
        "hazard": "Hand/finger amputation, cuts, chemical burns",
    },
    "Without Glass": {
        "event": "Struck by falling object or equipment, n.e.c.",
        "source": "Floor, n.e.c.",
        "hazard": "Eye injury from flying particles, chemical splash, dust",
    },
    "Without Mask": {
        "event": "Exposure to caustic, noxious, allergenic substance, unspecified",
        "source": "Nonclassifiable",
        "hazard": "Respiratory damage from dust, fumes, chemical vapors",
    },
    "Without Safety Shoes": {
        "event": "Struck by falling object or equipment, n.e.c.",
        "source": "Floor, n.e.c.",
        "hazard": "Foot crush injury, puncture wounds, electrical shock",
    },
    "Without Ear Protectors": {
        "event": "Exposure to noise",
        "source": "Machinery, unspecified",
        "hazard": "Hearing damage from prolonged noise exposure",
    },
}


def build_risk_prompt(detections: list[dict], osha_context: str, violation_risks: list[dict]) -> str:
    """Build the prompt for the risk scorer."""
    violations = [d for d in detections if "Without" in d.get("class_name", "")]

    # format per-violation risk info
    risk_summary = ""
    for vr in violation_risks:
        risk_summary += f"- {vr['violation']}: {vr['injury_prob']:.1%} severe injury probability — {vr['hazard']}\n"

    max_prob = max((vr["injury_prob"] for vr in violation_risks), default=0)

    return f"""You are a construction safety risk assessor.

Given the following PPE violations detected on a construction site and relevant OSHA regulations, 
provide a risk score from 1-10 and explain your reasoning.

## Detected Violations
{json.dumps(violations, indent=2)}

## Relevant OSHA Regulations
{osha_context}

## Injury Prediction Model (per violation)
{risk_summary}
Overall maximum severe injury probability: {max_prob:.1%}
Total violations detected: {len(violations)}

## Instructions
- Score 1-3: Minor (e.g., single missing glove, low-risk area)
- Score 4-5: Moderate (e.g., missing safety vest)
- Score 6-7: Serious (e.g., missing hard hat in active construction)
- Score 8-10: Critical (e.g., multiple missing PPE, fall protection absent)
- Consider BOTH the number of violations AND their individual severity

Respond in this exact JSON format:
{{"risk_score": <number 1-10>, "risk_reasoning": "<2-3 sentence explanation>", "osha_citations": ["<relevant OSHA section numbers>"]}}
"""


def risk_scorer_node(state: dict) -> dict:
    """LangGraph node: score risk from detections + OSHA context."""
    retriever = OSHARetriever()
    predictor = InjuryPredictor()

    # build query from ALL violation types
    violation_types = [d["class_name"] for d in state["detections"] if "Without" in d.get("class_name", "")]
    unique_violations = list(set(violation_types))
    query = f"PPE violations: {', '.join(unique_violations)}" if unique_violations else "general PPE requirements"

    osha_context = retriever.query(query)

    # calculate per-violation injury probabilities
    violation_risks = []
    for v_type in unique_violations:
        mapping = VIOLATION_EVENT_MAP.get(v_type, {
            "event": "Other", "source": "Nonclassifiable", "hazard": "General workplace hazard"
        })
        prob = predictor.predict(event_type=mapping["event"], source=mapping["source"])
        violation_risks.append({
            "violation": v_type,
            "injury_prob": prob,
            "hazard": mapping["hazard"],
        })

    prompt = build_risk_prompt(state["detections"], osha_context, violation_risks)

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    response = llm.invoke(prompt)

    try:
        result = json.loads(response.content)
    except json.JSONDecodeError:
        result = {"risk_score": 5, "risk_reasoning": response.content, "osha_citations": []}

    return {
        "osha_context": osha_context,
        "risk_score": result.get("risk_score", 5),
        "risk_reasoning": result.get("risk_reasoning", ""),
        "osha_citations": result.get("osha_citations", []),
    }
