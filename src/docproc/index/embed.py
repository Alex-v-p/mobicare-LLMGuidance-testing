from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class TfidfEmbedder:
    """Local, no-download embedding for benchmarking chunking.

    TF-IDF is not a semantic embedding, but it's good enough to compare
    chunking strategies in a deterministic way without pulling models.
    """

    max_features: int = 60_000
    ngram_range: tuple[int, int] = (1, 2)

    def fit_transform(self, texts: Sequence[str]) -> np.ndarray:
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            lowercase=True,
            strip_accents="unicode",
        )
        mat = self.vectorizer.fit_transform(texts)
        return mat.astype(np.float32)

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        mat = self.vectorizer.transform(texts)
        return mat.astype(np.float32)
