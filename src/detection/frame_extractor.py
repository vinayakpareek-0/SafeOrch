import cv2
from pathlib import Path


def extract_frames(video_path:str , fps:int=1)-> list[dict]:
    """Extract frames from video at given FPS.
    
    Returns list of {"frame_id": int, "timestamp": str, "frame": numpy array}
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video:{video_path}")

    video_fps =cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps/fps) # grab every nth frame
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames= []
    frame_count=0
    while True:
        ret,frame= cap.read()
        if not ret:
            break
        if frame_count % frame_interval ==0:
            seconds = frame_count / video_fps
            timestamp = f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"
            frames.append({
                "frame_id": len(frames),
                "timestamp": timestamp,
                "frame": frame,
            })
        frame_count += 1
    cap.release()
    return frames