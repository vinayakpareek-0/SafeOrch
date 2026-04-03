from typing import TypedDict
class SafetyState(TypedDict):
    # input from detection pipeline
    detections: list[dict]

    # from RAG retrieval
    osha_context: str

    # from Risk Scorer agent
    risk_score: float
    risk_reasoning: str
    
    # from Action Agent
    decision: str  # "alert" | "remediate" | "auto_stop"
    recommended_actions: list[str]
    osha_citations: list[str]