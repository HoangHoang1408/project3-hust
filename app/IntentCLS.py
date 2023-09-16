import json
import os
from collections import Counter

import numpy as np


# intent cls
class IntentClassifier:
    def __init__(self, embedder, data):
        self.data = data
        self.embedder = embedder

    def build(self):
        temp, self.metadata = [], []
        for intent_type in self.data:
            for question in intent_type["texts"]:
                temp.append(question)
                self.metadata.append(
                    {
                        "intent": intent_type["intent"],
                    }
                )
        self.embeddings = self.embedder.embed_documents(temp)

    def save_local(self, path):
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, "embeddings.npy"), self.embeddings)
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(self.metadata, f)
        with open(os.path.join(path, "data.json"), "w") as f:
            json.dump(self.data, f)

    @staticmethod
    def load_local(path, embedder):
        # check all paths exist
        paths = [
            os.path.join(path, "embeddings.npy"),
            os.path.join(path, "metadata.json"),
            os.path.join(path, "data.json"),
        ]
        for temp in paths:
            if not os.path.exists(temp):
                raise ValueError(f"Path {temp} does not exist")

        embeddings = np.load(os.path.join(path, "embeddings.npy"))
        with open(os.path.join(path, "metadata.json"), "r") as f:
            metadata = json.load(f)
        with open(os.path.join(path, "data.json"), "r") as f:
            data = json.load(f)

        intent_cls = IntentClassifier(embedder, data)
        intent_cls.embeddings = embeddings
        intent_cls.metadata = metadata
        return intent_cls

    def __call__(self, question, k=8):
        question_repr = self.embedder.embed_query(question)
        scores = np.dot(self.embeddings, question_repr)
        top_k = np.argsort(scores)[::-1][:k]
        counter = Counter([self.metadata[x]["intent"] for x in top_k])
        return counter.most_common(1)[0][0]
