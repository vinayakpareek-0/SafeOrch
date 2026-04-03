from langgraph.graph import StateGraph, END
from src.agents.state import SafetyState
from src.agents.risk_scorer import risk_scorer_node
from src.agents.action_agent import action_agent_node

def build_safety_graph():
    """Build the Langgraph agent pipeline"""
    graph = StateGraph(SafetyState)

    graph.add_node("risk_scorer", risk_scorer_node)
    graph.add_node("action_agent", action_agent_node)
    graph.set_entry_point("risk_scorer")
    graph.add_edge("risk_scorer", "action_agent")
    graph.add_edge("action_agent",END)

    return graph.compile()

if __name__=="__main__":
    import json

    # test eith sample detection
    sample_detections= [
        {"class_name":"Without Helmet", "confidence": 0.85, "bbox":[100,200,300,400]},
        {"class_name":"Without Vest", "confidence": 0.72, "bbox":[150,1080,350,420]},
        {"class_name":"Person", "confidence": 0.91, "bbox":[80,150,380,450]},
        {"class_name":"Without Gloves", "confidence": 0.91, "bbox":[80,150,380,450]},
    ]

    graph=build_safety_graph()
    result=graph.invoke({"detections":sample_detections})
    print(json.dumps({
        "risk_score": result["risk_score"],
        "risk_reasoning": result["risk_reasoning"],
        "decision": result["decision"],
        "recommended_actions": result["recommended_actions"],
        "osha_citations": result["osha_citations"],
    }, indent=2))