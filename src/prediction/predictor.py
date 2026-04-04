import joblib
import pandas as pd
from pathlib import Path

class InjuryPredictor:
    """Predict injury severity probability from event context."""

    def __init__(self):
        model_dir=Path(__file__).parent.parent.parent/"models"
        self.model = joblib.load(model_dir/"injury_predictor.pkl")
        self.encoders = joblib.load(model_dir/"label_encoders.pkl")

    def predict(self , event_type:str , source:str, naics_code :int=0 )-> float:
        """Predict injury probability."""
        inputs = {
            "EventTitle":event_type,
            "SourceTitle":source,
            "Primary NAICS":str(naics_code),
        }

        encoded= {}

        for col,val in inputs.items():
            le =self.encoders[col]
            if val in le.classes_:
                encoded[col]=le.transform([val])[0]
            else:
                encoded[col]=0

        prob = self.model.predict_proba(pd.DataFrame([encoded]))[0][1]
        return round(prob,4)

if __name__ == "__main__":
    p =InjuryPredictor()
    prob=p.predict(
        event_type="Other fall to lower 6 to 10 feet",
        source="Ladders, unspecified",
    )
    print(f"Injury severity probability: {prob:.2%}")