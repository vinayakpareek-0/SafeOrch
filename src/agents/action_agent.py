import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()


def action_agent_node(state: dict) -> dict:
    """LangGraph node: decide action based on risk score."""
    prompt = f"""You are a construction safety action planner.

Based on the risk assessment below, decide the appropriate action and provide specific recommendations.

## Risk Assessment
- Risk Score: {state['risk_score']}/10
- Reasoning: {state['risk_reasoning']}
- OSHA Citations: {', '.join(state.get('osha_citations', []))}

## Action Thresholds
- Score 3-5: "alert" — Notify supervisor, log violation
- Score 6-7: "remediate" — Stop task, provide correct PPE, retrain before resuming
- Score 8-10: "auto_stop" — Immediate work stoppage, incident report, mandatory safety stand-down

## Instructions
Choose one action and provide 2-4 specific recommended steps.

Respond in this exact JSON format:
{{"decision": "alert|remediate|auto_stop", "recommended_actions": ["action 1", "action 2", "action 3"]}}
"""

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    response = llm.invoke(prompt)

    try:
        result = json.loads(response.content)
    except json.JSONDecodeError:
        # fallback based on score
        score = state.get("risk_score", 5)
        if score >= 8:
            decision = "auto_stop"
        elif score >= 6:
            decision = "remediate"
        else:
            decision = "alert"
        result = {"decision": decision, "recommended_actions": [response.content]}

    return {
        "decision": result.get("decision", "alert"),
        "recommended_actions": result.get("recommended_actions", []),
    }
