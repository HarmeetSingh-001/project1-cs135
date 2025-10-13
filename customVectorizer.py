from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

class CustomFilteredTfidfVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, min_count=1, ngram_range=(1,2), max_features=5000, max_df=10000,min_df=1,preprocessor=None):
        self.min_count = min_count
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.max_df = max_df
        self.min_df = min_df 
        self.preprocessor = preprocessor
        self.vectorizer_ = None
    
    def tokenize_text(self, raw_text):
        list_of_tokens = raw_text.split()
        for i in range(len(list_of_tokens)):
            cur_token = list_of_tokens[i]
            for punc in ['?', '!', '_', '.', ',', '"', '/']:
                cur_token = cur_token.replace(punc, "")
            list_of_tokens[i] = cur_token.lower()
        return list_of_tokens
    
    def fit(self, X, y=None):
        from collections import Counter
        
        # Count token frequency across all documents
        token_counts = Counter()
        for doc in X:
            if self.preprocessor:
                doc = self.preprocessor(doc)
            tokens = self.tokenize_text(doc)
            token_counts.update(tokens)
        
        # Filter tokens by min_count threshold
        filtered_vocab = [token for token, count in token_counts.items() if count > self.min_count]
        
        # Create the vectorizer with fixed filtered vocabulary
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

        print(f"[INFO] Fitted TF-IDF with min_count={self.min_count}, min_df={self.min_df}, max_df={self.max_df} "
          f"→ Vocabulary size: {len(self.vectorizer_.vocabulary_)}")

        return self
    
    def transform(self, X):
        return self.vectorizer_.transform(X)
