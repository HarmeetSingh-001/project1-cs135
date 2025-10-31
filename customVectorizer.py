from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from nltk.stem import PorterStemmer


class CustomFilteredTfidfVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, 
                 min_count=0,
                 max_count=float('inf'),
                 min_length=1,             
                 ngram_range=(1, 1),
                 max_features=5000,
                 max_df=1.0,
                 min_df=1,
                 use_stemming=False,
                 preprocessor=None):
        
        self.min_count = min_count
        self.max_count = max_count
        self.min_length = min_length    
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.max_df = max_df
        self.min_df = min_df
        self.use_stemming = use_stemming
        self.preprocessor = preprocessor
        self.vectorizer_ = None
        self.stemmer = PorterStemmer() if use_stemming else None

    def tokenize_text(self, raw_text):
        tokens = raw_text.split()
        cleaned_tokens = []
        for token in tokens:
            for punc in ['?', '!', '_', '.', ',', '"', '/']:
                token = token.replace(punc, "")
            token = token.lower().strip()
            if len(token) >= self.min_length:  
                cleaned_tokens.append(token)
            
        if self.use_stemming:
            tokens = [self.stemmer.stem(tok) for tok in tokens]
        return cleaned_tokens

    def fit(self, X, y=None):
        token_counts = Counter()

        for doc in X:
            if self.preprocessor:
                doc = self.preprocessor(doc)
            tokens = self.tokenize_text(doc)
            token_counts.update(tokens)

        # Filter tokens by both frequency and length
        filtered_vocab = [
            token for token, count in token_counts.items()
            if (self.min_count < count < self.max_count and len(token) >= self.min_length)
        ]

        # Initialize internal TF–IDF vectorizer
        self.vectorizer_ = TfidfVectorizer(
            tokenizer=self.tokenize_text,
            lowercase=False,
            token_pattern=None,
            vocabulary=filtered_vocab,
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            max_df=self.max_df,
            min_df=self.min_df,
            preprocessor=self.preprocessor
        )

        self.vectorizer_.fit(X)
        return self

    def transform(self, X):
        return self.vectorizer_.transform(X)

    def get_params(self, deep=True):
        return {
            'min_count': self.min_count,
            'max_count': self.max_count,
            'min_length': self.min_length,      
            'ngram_range': self.ngram_range,
            'max_features': self.max_features,
            'max_df': self.max_df,
            'min_df': self.min_df,
            'use_stemming': self.use_stemming,
            'preprocessor': self.preprocessor,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        if 'use_stemming' in params:
            self.stemmer = PorterStemmer() if params['use_stemming'] else None
        return self
