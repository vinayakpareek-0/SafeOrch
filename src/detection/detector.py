from ultralytics import YOLO
from .schemas import Detection, FrameResult

class PPEDetector:
    """Wrapper aroung YOLO model for PPE detection."""
    def __init__(self , model_path: str , confidence:float=0.5):
        self.model = YOLO(model_path)
        self.confidence = confidence

    def detect_frame(self, frame, frame_id:int , timestamp:str ) -> FrameResult:
        """Run detection on single frame"""
        result = self.model(frame , conf = self.confidence, verbose=False)[0]
        detections=[]

        for box in result.boxes:
            detections.append(Detection(
                class_name = result.names[int(box.cls[0])],
                confidence= round(float(box.conf[0]),3),
                bbox = [round(float(c),1) for c in box.xyxy[0]],
            ))
        return FrameResult(
            frame_id =frame_id,
            timestamp = timestamp,
            detections=detections,
        )