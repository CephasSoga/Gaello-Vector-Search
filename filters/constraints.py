import time
import subprocess
from typing import List, Any
from dataclasses import dataclass

import spacy
from spacy import Language
from spacy.tokens import Doc

class Parser:

    def minimize(self, nlp: Language, doc: Doc) -> str:
        """Remove stop words and non-alphabetic tokens from the document."""
        stop_words = nlp.Defaults.stop_words
        min_qx = [token.text for token in doc if token.is_alpha and token.text.lower() not in stop_words]
        return "".join(min_qx)
    
    def extract_kwds(self, doc: Doc) -> List[str]:
        """Extract nouns from the document."""
        nouns = [token.text for token in doc if token.pos_ == 'NOUN']
        return nouns

    def similarity(self, *kwds: str) -> float:
        """Calculate the similarity score based on the intersection of keyword sets."""
        if not kwds:
            return 0.0
        # Convert the first list to a set
        x = set(kwds[0])
        # Compute the intersection with all other lists
        for kw in kwds[1:]:
            y = set(kw)
            x = x.intersection(y)
        # Compute the similarity score
        # Return 0 if all lists are empty
        if len(kwds) == 0 or min(len(kw) for kw in kwds) == 0:
            return 0.0
        # Compute the minimum length of the input lists
        min_len = min(len(kw) for kw in kwds)
        
        return len(x) / min_len
    

@dataclass
class Filter:
    ctx: List[str]
    threshold: float
    model: str

    def __post_init__(self):
        try:
            _ = spacy.load(self.model)
        except OSError:
            subprocess.run(['python', '-m', 'spacy', 'download', self.model])
            time.sleep(20)
        finally:
            self.nlp = spacy.load(self.model)

        self.parser: Parser = Parser()

    def __call__(self, target: str) -> Any:
        kwds_list = []

        tx_doc = self.nlp(target)
        min_tx = self.parser.minimize(self.nlp, tx_doc)

        min_tx_doc = self.nlp(min_tx)
        tx_kwds = self.parser.extract_kwds(min_tx_doc)

        for ctx in self.ctx:
            doc = self.nlp(ctx)
            min_qx = self.parser.minimize(self.nlp, doc)

            min_doc = self.nlp(min_qx)
            kwds = self.parser.extract_kwds(doc)
            
            kwds_list.append(kwds)
        
        return [kwds for kwds in kwds_list if self.parser.similarity(tx_kwds, kwds) >= self.threshold]
            

    