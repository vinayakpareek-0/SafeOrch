from dataclasses import dataclass, field

@dataclass 
class Detection:
    """Single detected object in a frame"""
    class_name : str
    confidence : float
    bbox: list 

@dataclass
class FrameResult:
    """all detections from one frame"""
    frame_id: int
    timestamp:str  # "00:00:05"
    detections: list[Detection] = field(default_factory=list)

@dataclass
class PipelineResult:
    """full pipeline output for a video/ img"""
    source: str
    total_frames:int
    frames_analyzed:int
    frame_results: list[FrameResult] = field(default_factory=list)

