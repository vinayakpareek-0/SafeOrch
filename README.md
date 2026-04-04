# SafeOrch: Intelligent Safety Orchestration Platform

An end-to-end safety monitoring system that processes construction site footage through a multi-agent pipeline to detect PPE violations, assess risk using OSHA regulations, predict injury severity, and generate actionable incident reports.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Pipeline Details](#pipeline-details)
- [Configuration](#configuration)
- [License](#license)

## Overview

SafeOrch addresses the critical challenge of workplace safety compliance in construction environments. The platform takes an image, video, or live webcam feed as input and runs it through a multi-stage pipeline:

1. **Detection**: A fine-tuned YOLO model identifies PPE compliance and violations across 15 classes.
2. **Retrieval**: Relevant OSHA regulations are retrieved from a vector database to provide regulatory context.
3. **Prediction**: An XGBoost model trained on 100K+ OSHA incident records predicts the probability of severe injury for each violation type.
4. **Risk Scoring**: A LangGraph agent synthesizes detections, OSHA context, and injury predictions to produce a risk score (1-10) with reasoning.
5. **Action Decision**: A second agent determines the appropriate response (alert, remediate, or auto-stop) with specific recommended actions.
6. **Reporting**: A downloadable PDF report is generated with annotated frames, violations, risk assessment, OSHA citations, and recommended actions.

## Architecture

```
Input (Image / Video / Webcam)
        |
        v
+------------------+
|  YOLO Detection  |  -- Fine-tuned YOLOv11n on PPE dataset (15 classes)
+------------------+
        |
        v
+------------------+     +------------------+
|  RAG Retriever   |---->|   ChromaDB       |  -- OSHA regulation vectors
+------------------+     +------------------+
        |
        v
+------------------+     +------------------+
|  Risk Scorer     |---->| Injury Predictor |  -- XGBoost on OSHA incident data
|  (LLM Agent)     |     +------------------+
+------------------+
        |
        v
+------------------+
|  Action Agent    |  -- Decides: alert / remediate / auto_stop
|  (LLM Agent)     |
+------------------+
        |
        v
+------------------+
|  PDF Report      |  -- Incident report with citations and actions
+------------------+
```

## Tech Stack

| Layer                | Technology                                  |
| -------------------- | ------------------------------------------- |
| Object Detection     | YOLOv11n (Ultralytics), fine-tuned on PPE   |
| Agent Orchestration  | LangGraph (StateGraph)                      |
| LLM Inference        | Groq API (Llama 3.3 70B Versatile)          |
| RAG / Vector Store   | LangChain + ChromaDB + HuggingFace Embeddings |
| Embedding Model      | all-MiniLM-L6-v2 (sentence-transformers)    |
| Injury Prediction    | XGBoost + scikit-learn                      |
| Frontend             | Streamlit                                   |
| Report Generation    | fpdf2                                       |
| Computer Vision      | OpenCV                                      |
| Configuration        | PyYAML                                      |

## Project Structure

```
SafeOrch/
|-- app/
|   |-- streamlit_app.py          # Main Streamlit application
|
|-- config/
|   |-- settings.yaml             # Centralized configuration
|
|-- data/
|   |-- osha_docs/                # OSHA regulation text files (5 standards)
|   |-- raw/                      # Raw datasets (gitignored)
|   |-- test_data/                # Test images and videos (gitignored)
|
|-- models/
|   |-- yolo_ppe_v11/             # Fine-tuned YOLO model and training artifacts
|   |-- injury_predictor.pkl      # Trained XGBoost model (gitignored)
|   |-- label_encoders.pkl        # Fitted label encoders (gitignored)
|
|-- notebooks/
|   |-- 01_yolo_ppe_training.ipynb    # YOLO fine-tuning on Roboflow PPE dataset
|   |-- 02_osha_data_analysis.ipynb   # OSHA data EDA and XGBoost training
|
|-- src/
|   |-- detection/
|   |   |-- schemas.py            # Data classes: Detection, FrameResult, PipelineResult
|   |   |-- detector.py           # PPEDetector wrapper around YOLO
|   |   |-- frame_extractor.py    # Video frame extraction at configurable FPS
|   |   |-- pipeline.py           # End-to-end detection pipeline
|   |
|   |-- rag/
|   |   |-- ingest.py             # Chunk and embed OSHA docs into ChromaDB
|   |   |-- retriever.py          # Query ChromaDB for relevant regulation context
|   |
|   |-- agents/
|   |   |-- state.py              # SafetyState TypedDict (shared agent state)
|   |   |-- risk_scorer.py        # Risk scoring agent with per-violation mapping
|   |   |-- action_agent.py       # Action decision agent
|   |   |-- graph.py              # LangGraph pipeline wiring
|   |
|   |-- prediction/
|   |   |-- predictor.py          # InjuryPredictor: loads XGBoost for inference
|   |
|   |-- reporting/
|       |-- pdf_generator.py      # PDF incident report generation
|
|-- requirements.txt
|-- .env                          # API keys (gitignored)
|-- .gitignore
|-- .gitattributes
```

## Setup

### Prerequisites

- Python 3.10+
- Git

### Installation

1. Clone the repository:

```bash
git clone https://github.com/vinayakpareek-0/SafeOrch.git
cd SafeOrch
```

2. Create and activate a virtual environment:

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up environment variables. Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key
ROBOFLOW_API_KEY=your_roboflow_api_key
```

You can get a free Groq API key at [console.groq.com](https://console.groq.com).

5. Ingest OSHA documents into ChromaDB:

```bash
python -m src.rag.ingest
```

6. Train the injury prediction model (or use the provided notebook):

```bash
# Option A: Run the notebook
jupyter notebook notebooks/02_osha_data_analysis.ipynb

# Option B: The model files are generated by the notebook
# and saved to models/injury_predictor.pkl and models/label_encoders.pkl
```

Note: The YOLO model weights (`models/yolo_ppe_v11/weights/best.pt`) must be present. These are generated by the training notebook (`01_yolo_ppe_training.ipynb`) using the Roboflow PPE dataset.

## Usage

### Streamlit Application

```bash
streamlit run app/streamlit_app.py
```

The application provides two input modes:

- **Upload**: Drag and drop an image (JPG, PNG) or video (MP4, AVI) for batch analysis.
- **Webcam Snapshot**: Capture a photo directly from your webcam for instant analysis.

Results are displayed as a split view with the annotated detection frame on the left and the agent analysis (risk score, decision, reasoning, OSHA citations, recommended actions) on the right. A downloadable PDF report is available when violations are detected.

### Detection Pipeline (CLI)

```bash
python -m src.detection.pipeline
```

Processes video files frame-by-frame and outputs structured JSON detection results.

### Agent Pipeline (CLI)

```bash
python -m src.agents.graph
```

Runs the full agent pipeline with sample detections to verify the risk scoring and action decision flow.

### RAG Retriever (CLI)

```bash
python -m src.rag.retriever
```

Queries the ChromaDB vector store with a sample query to verify OSHA document retrieval.

## Model Performance

### YOLO PPE Detection

| Metric       | Value   |
| ------------ | ------- |
| Model        | YOLOv11n |
| mAP@0.5      | 0.673   |
| Classes      | 15 (8 PPE present + 7 PPE absent) |
| Training Data| Roboflow PPE dataset |
| Image Size   | 640x640 |

### Injury Severity Prediction

| Metric    | Value   |
| --------- | ------- |
| Model     | XGBoost |
| AUC-ROC   | 0.922   |
| Accuracy  | 86%     |
| Recall (severe) | 88% |
| Dataset   | OSHA Severe Injury Reports (102K records) |
| Features  | EventTitle, SourceTitle, Primary NAICS |

## Pipeline Details

### Detection Classes

The YOLO model detects the following PPE classes:

**Compliant (PPE present):** Helmet, Vest, Gloves, Glasses, Mask, Safety Shoes, Ear Protectors, Person

**Violations (PPE absent):** Without Helmet, Without Vest, Without Glove, Without Glass, Without Mask, Without Safety Shoes, Without Ear Protectors

### Risk Scoring

Each violation type is mapped to a specific OSHA event category for injury prediction:

| Violation             | OSHA Event Mapping                          | Primary Hazard                |
| --------------------- | ------------------------------------------- | ----------------------------- |
| Without Helmet        | Struck by falling object                    | Head injury, electrical shock |
| Without Vest          | Struck by powered vehicle                   | Low visibility                |
| Without Glove         | Caught in running machinery                 | Amputation, cuts              |
| Without Glass         | Struck by falling object                    | Eye injury, chemical splash   |
| Without Mask          | Exposure to caustic substance               | Respiratory damage            |
| Without Safety Shoes  | Struck by falling object                    | Foot crush, puncture wounds   |
| Without Ear Protectors| Exposure to noise                           | Hearing damage                |

### Action Decision Thresholds

| Risk Score | Decision    | Response                                    |
| ---------- | ----------- | ------------------------------------------- |
| 3 to 5     | Alert       | Notify supervisor, log violation            |
| 6 to 7     | Remediate   | Stop task, provide correct PPE, retrain     |
| 8 to 10    | Auto Stop   | Immediate work stoppage, incident report    |

### OSHA Standards in RAG

The following OSHA regulations are embedded in the vector store:

- **1926.95**: Criteria for Personal Protective Equipment
- **1926.100**: Head Protection
- **1926.102**: Eye and Face Protection
- **1910.132**: General PPE Requirements
- **1926.501**: Duty to Have Fall Protection

## Configuration

All configuration is centralized in `config/settings.yaml`:

```yaml
detection:
  model_path: "models/yolo_ppe_v11/weights/best.pt"
  confidence_threshold: 0.5
  iou_threshold: 0.7
  imgsz: 640
  frame_extraction_fps: 1

rag:
  chroma_db_path: "chroma_db/"
  chunk_size: 500
  chunk_overlap: 100
  top_k: 5

agents:
  llm_provider: "groq"
  groq_model: "llama-3.3-70b-versatile"
  risk_threshold:
    alert: [3, 5]
    remediate: [6, 7]
    auto_stop: [8, 10]
```

## License

This project is developed as a capstone project. See the repository for license details.
