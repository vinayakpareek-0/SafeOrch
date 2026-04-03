import json
import yaml
from pathlib import Path
from dataclasses import asdict
from .frame_extractor import extract_frames
from .detector import PPEDetector
from .schemas import PipelineResult

def load_config():
    """load settings from config/settings.yaml"""
    config_path = Path(__file__).parent.parent/"config"/"settings.yaml"
    with open(config_path ) as f:
        return yaml.safe_load(f)

def run_pipeline(video_path:str)->dict:
    """Run full detection pipeline on video file"""
    config=load_config()
    det_config = config["detection"]

    frames=extract_frames(video_path,fps=det_config["frame_extraction_fps"])

    model_path= Path(__file__).parent.parent / det_config["model_path"]
    conf= det_config["confidence_threshold"]
    detector = PPEDetector(str(model_path), confidence=conf)

    frame_results=[]
    for f in frames:
        result=detector.detect_frame(f["frame"],f["frame_id"],f["timestamp"]) 
        frame_results.append(result)

    pipeline_result = PipelineResult(
        source=video_path,
        total_frames = len(frames ) * det_config["frame_extraction_fps"],
        frames_analyzed= len(frames),
        frame_results = frame_results,
    )
    return asdict(pipeline_result)

if __name__ =="__main__":
    import sys
    if len(sys.argv)<2: 
        print("Usage:python -m src.detection.pipeline <video_or_image_path>")
        sys.exit(1)
    
    result= run_pipeline(sys.argv[1])
    print(json.dumps(result, indent=2))