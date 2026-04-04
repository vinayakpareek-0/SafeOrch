import io
import cv2
import tempfile
from datetime import datetime
from fpdf import FPDF


def generate_report(annotated_frame, detections: list[dict], agent_result: dict) -> bytes:
    """Generate a PDF safety incident report. Returns PDF as bytes."""
    pdf = FPDF()
    pdf.add_page()

    # Header
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 12, "SafeOrch Safety Incident Report", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(5)

    # Annotated Frame
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        cv2.imwrite(tmp.name, annotated_frame)
        pdf.image(tmp.name, x=15, w=180)
    pdf.ln(5)

    # Risk Summary
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Risk Assessment", new_x="LMARGIN", new_y="NEXT")

    score = agent_result.get("risk_score", "N/A")
    decision = agent_result.get("decision", "N/A").upper()

    pdf.set_font("Helvetica", "", 11)
    pdf.cell(50, 8, f"Risk Score: {score}/10")
    pdf.cell(0, 8, f"Decision: {decision}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Reasoning:", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, agent_result.get("risk_reasoning", "N/A"))
    pdf.ln(3)

    # Violations
    violations = [d for d in detections if "Without" in d.get("class_name", "")]
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, f"Violations Detected ({len(violations)})", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(80, 8, "Violation", border=1)
    pdf.cell(40, 8, "Confidence", border=1)
    pdf.cell(0, 8, "Bounding Box", border=1, new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 10)
    for v in violations:
        pdf.cell(80, 7, v["class_name"], border=1)
        pdf.cell(40, 7, f"{v['confidence']:.0%}", border=1)
        pdf.cell(0, 7, str(v.get("bbox", "")), border=1, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # OSHA Citations 
    citations = agent_result.get("osha_citations", [])
    if citations:
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "OSHA Citations", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        for c in citations:
            pdf.cell(0, 7, f"  - OSHA {c}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

    # Recommended Actions
    actions = agent_result.get("recommended_actions", [])
    if actions:
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Recommended Actions", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        for i, action in enumerate(actions, 1):
            pdf.multi_cell(0, 6, f"{i}. {action}")
            pdf.ln(1)

    return bytes(pdf.output())
