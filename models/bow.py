from pathlib import Path

import joblib


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT /
    "train" /
    "saved_models" /
    "bow_pipeline.joblib"
)


class BowModel:

    def __init__(self):
        self.pipeline = joblib.load(MODEL_PATH)

    def predict(self, text: str):

        prediction = self.pipeline.predict([text])[0]

        probabilities = self.pipeline.predict_proba([text])[0]

        return {
            "prediction": prediction,
            "probabilities": {
                classe: float(prob)
                for classe, prob in zip(
                    self.pipeline.classes_,
                    probabilities
                )
            }
        }