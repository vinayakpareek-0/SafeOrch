import json
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from src.rag.retriever import OSHARetriever
from src.prediction.predictor import InjuryPredictor

load_dotenv()


def build_risk_prompt(detections: list[dict], osha_context: str, injury_prob: float) -> str:    
    """Build the prompt for the risk scorer."""
    violations = [d for d in detections if "Without" in d.get("class_name", "")]
    
    return f"""You are a construction safety risk assessor.

Given the following PPE violations detected on a construction site and relevant OSHA regulations, 
provide a risk score from 1-10 and explain your reasoning.

## Detected Violations
{json.dumps(violations, indent=2)}

## Relevant OSHA Regulations
{osha_context}

## Injury Prediction Model
Based on historical OSHA incident data, the predicted probability of severe injury (amputation/loss) for this scenario is:{injury_prob:.1%}

## Instructions
- Score 1-3: Minor (e.g., single missing glove, low-risk area)
- Score 4-5: Moderate (e.g., missing safety vest)
- Score 6-7: Serious (e.g., missing hard hat in active construction)
- Score 8-10: Critical (e.g., multiple missing PPE, fall protection absent)

Respond in this exact JSON format:
{{"risk_score": <number 1-10>, "risk_reasoning": "<2-3 sentence explanation>", "osha_citations": ["<relevant OSHA section numbers>"]}}
"""


def risk_scorer_node(state: dict) -> dict:
    """LangGraph node: score risk from detections + OSHA context."""
    retriever = OSHARetriever()
    predictor = InjuryPredictor()

    # build query from violation types
    violation_types = [d["class_name"] for d in state["detections"] if "Without" in d.get("class_name", "")]
    query = f"PPE violations: {', '.join(set(violation_types))}" if violation_types else "general PPE requirements"
    
    osha_context = retriever.query(query)
    event_type = "Struck by falling object or equipment, n.e.c." if violation_types else "Other"
    injury_prob = predictor.predict(event_type=event_type, source="Nonclassifiable")
    prompt = build_risk_prompt(state["detections"], osha_context, injury_prob)

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
