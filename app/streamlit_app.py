import streamlit as st
import cv2
import json
import tempfile
import numpy as np
from pathlib import Path
from PIL import Image

# add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection.detector import PPEDetector
from src.agents.graph import build_safety_graph
from src.reporting.pdf_generator import generate_report

st.set_page_config(page_title="SafeOrch", page_icon="🦺", layout="wide")

#  cached resources 
@st.cache_resource
def load_detector():
    model_path = Path(__file__).parent.parent / "models" / "yolo_ppe_v11" / "weights" / "best.pt"
    return PPEDetector(str(model_path), confidence=0.35)

@st.cache_resource
def load_graph():
    return build_safety_graph()


def annotate_frame(frame, detections):
    """Draw bounding boxes on frame."""
    annotated = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(c) for c in det["bbox"]]
        is_violation = "Without" in det["class_name"]
        color = (0, 0, 255) if is_violation else (0, 255, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{det['class_name']} {det['confidence']:.0%}"
        cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return annotated


def process_frame(frame, detector, graph):
    """Run detection + agents on a single frame."""
    result = detector.detect_frame(frame, frame_id=0, timestamp="00:00")
    detections = [{"class_name": d.class_name, "confidence": d.confidence, "bbox": d.bbox} for d in result.detections]

    annotated = annotate_frame(frame, detections)

    # run agents if violations found
    violations = [d for d in detections if "Without" in d["class_name"]]
    agent_result = None
    if violations:
        agent_result = graph.invoke({"detections": detections})

    return annotated, detections, agent_result


def show_results(annotated, detections, agent_result):
    """Display detection + agent results."""
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📸 Detection Results")
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

        violations = [d for d in detections if "Without" in d["class_name"]]
        safe = [d for d in detections if "Without" not in d["class_name"]]
        st.metric("Violations", len(violations))
        st.metric("PPE Detected", len(safe))

    with col2:
        st.subheader("🤖 Agent Analysis")
        if agent_result:
            score = agent_result["risk_score"]
            color = "🔴" if score >= 8 else "🟡" if score >= 5 else "🟢"
            st.metric("Risk Score", f"{color} {score}/10")
            st.info(f"**Decision:** {agent_result['decision'].upper()}")
            st.write("**Reasoning:**", agent_result["risk_reasoning"])
            st.write("**Recommended Actions:**")
            for action in agent_result.get("recommended_actions", []):
                st.write(f"- {action}")
            st.write("**OSHA Citations:**", ", ".join(agent_result.get("osha_citations", [])))
        else:
            st.success("✅ No violations detected!")
        if agent_result:
            pdf_bytes = generate_report(annotated, detections, agent_result)
            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_bytes,
                file_name=f"safeorch_report_{int(__import__('time').time())}.pdf",
                mime="application/pdf",
            )


#  MAIN APP 
st.title("🦺 SafeOrch — Safety Orchestration Platform")
st.caption("Batch-process safety footage through a multi-agent pipeline")

detector = load_detector()
graph = load_graph()

tab1, tab2 = st.tabs(["📁 Upload Image/Video", "📷 Webcam Snapshot"])

#  TAB 1: Upload 
with tab1:
    uploaded = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4", "avi"])

    if uploaded:
        if uploaded.type.startswith("image"):
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            with st.spinner("Analyzing..."):
                annotated, detections, agent_result = process_frame(frame, detector, graph)
            show_results(annotated, detections, agent_result)

        elif uploaded.type.startswith("video"):
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            cap = cv2.VideoCapture(tmp_path)
            ret, frame = cap.read()
            cap.release()

            if ret:
                with st.spinner("Analyzing first frame..."):
                    annotated, detections, agent_result = process_frame(frame, detector, graph)
                show_results(annotated, detections, agent_result)

#  TAB 2: Webcam
with tab2:
    camera_input = st.camera_input("Take a photo")

    if camera_input:
        file_bytes = np.asarray(bytearray(camera_input.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        with st.spinner("Analyzing..."):
            annotated, detections, agent_result = process_frame(frame, detector, graph)
        show_results(annotated, detections, agent_result)

with st.sidebar:
    st.header("⚙️ Settings")
    confidence = st.slider("Detection Confidence", 0.1, 1.0, 0.3, 0.05)
    detector.model.conf = confidence