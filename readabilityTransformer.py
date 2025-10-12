from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import textstat

class ReadabilityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

	#Takes raw text and gets numerical features from it based off of various metrics
    def transform(self, X):
        features = []
        for text in X:
            features.append([
                textstat.flesch_kincaid_grade(text),
                textstat.smog_index(text),
                textstat.avg_sentence_length(text),
                textstat.syllable_count(text) / max(1, len(text.split())),
                textstat.lexicon_count(text, removepunct=True) / max(1, len(text.split()))
            ])
        return np.array(features)