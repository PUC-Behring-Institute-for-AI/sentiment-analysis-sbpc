from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

import torch


MODEL_NAME = "jvomiranda/BERTimbau-Sent-Analysis"


class BertimbauModel:

    def __init__(self):

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME
        )

        self.model = (
            AutoModelForSequenceClassification
            .from_pretrained(MODEL_NAME)
            .to(self.device)
        )

        self.model.eval()

    @torch.no_grad()
    def predict(self, text: str):

        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128
        )

        encoding = {
            key: value.to(self.device)
            for key, value in encoding.items()
        }

        outputs = self.model(**encoding)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )[0]

        prediction = torch.argmax(
            probabilities
        ).item()

        tokens = self.tokenizer.convert_ids_to_tokens(
            encoding["input_ids"][0]
        )

        token_ids = (
            encoding["input_ids"][0]
            .cpu()
            .tolist()
        )

        return {

            "prediction": prediction,

            "probabilities": {

                0: float(probabilities[0]),

                1: float(probabilities[1])

            },

            "tokens": tokens,

            "token_ids": token_ids

        }