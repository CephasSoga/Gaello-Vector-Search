import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Any
from dataclasses import dataclass

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from functools import lru_cache

from vector_search.utils.logs import timer


def download_nltk_data():
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')


class Parser:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    @lru_cache(maxsize=1024)  # Cache results for repeated calls
    @timer()
    def minimize(self, text: str) -> str:
        """Remove stop words and non-alphabetic tokens from the text."""
        words = word_tokenize(text)
        min_qx = [word for word in words if word.isalpha() and word.lower() not in self.stop_words]
        return " ".join(min_qx)

    @lru_cache(maxsize=1024)
    @timer()
    def extract_kwds(self, text: str) -> List[str]:
        """Extract nouns from the text."""
        words = word_tokenize(text)
        tagged_words = pos_tag(words)
        return [word for word, pos in tagged_words if pos.startswith('NN')]

    @timer()
    def similarity(self, *kwds: str) -> float:
        """Calculate the similarity score based on the intersection of keyword sets."""
        if not kwds:
            return 0.0
        if len(kwds) == 1:
            return 1.0
        if any(len(kw) == 0 for kw in kwds) and len(kwds) == 2:
            return 0.0
        x = set(kwds[0])
        for kw in kwds[1:]:
            x.intersection_update(kw)
        if not x:
            return 0.0
        min_len = min(len(kw) for kw in kwds if kw)
        return len(x) / min_len if min_len else 0.0


@dataclass
class Filter:
    ctx: List[str]
    threshold: float
    parser: Parser

    @timer()
    def __call__(self, target: str) -> Any:
        s = time.perf_counter()

        # Preprocess the target once
        min_tx = self.parser.minimize(target)
        tx_kwds = self.parser.extract_kwds(min_tx)

        # Function to process each context in parallel
        def process_context(idx, ctx):
            min_qx = self.parser.minimize(ctx)
            kwds = self.parser.extract_kwds(min_qx)
            return idx, kwds, ctx

        # Use parallel processing to minimize and extract keywords
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_context, idx, ctx) for idx, ctx in enumerate(self.ctx)]
            extracted_kwds = {str(idx): {'kwds': kwds, 'c': ctx} for idx, kwds, ctx in [f.result() for f in futures]}

        # Filter results based on similarity
        results = [v['c'] for v in extracted_kwds.values() if self.parser.similarity(tx_kwds, v['kwds']) >= self.threshold]

        e = time.perf_counter()
        print(f"Time for filtering: {e-s} seconds")
        return results


if __name__ == "__main__":
    download_nltk_data()
    f = Filter(ctx=["hello world", "hello universe"], threshold=0.5, parser=Parser())
    print(f("hello universe"))
