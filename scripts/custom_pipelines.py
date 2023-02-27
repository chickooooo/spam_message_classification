import pandas as pd
import spacy
from sklearn.base import BaseEstimator, TransformerMixin


class TextCleaner(BaseEstimator, TransformerMixin):
    """Cleans raw text data
    """

    def __init__(self):
        # load large english library
        self.__nlp = spacy.load("en_core_web_lg")

    def fit(self, X: pd.Series, y=None):
        return self
    
    def transform(self, X: pd.Series, y=None) -> pd.Series:
        # return cleaned text
        return X.apply(lambda x: self.__remove_junk(self.__nlp(x)))

    def __remove_junk(self, doc) -> str:
        # will hold cleaned text
        cleaned = ""

        # iterating through all the tokens
        for token in doc:
            # junk
            if token.is_space or token.is_punct or token.is_stop:
                pass
            # data
            else:
                # add base form of token
                cleaned += ' '
                cleaned += token.lemma_
        
        # return clean text
        return cleaned


class TargetEncoder(BaseEstimator, TransformerMixin):
    """Encodes target into 0 & 1
    """

    def __init__(self):
        pass

    def fit(self, X: pd.Series, y=None):
        return self
    
    def transform(self, X: pd.Series, y=None) -> pd.Series:
        # spam -> 1
        # ham -> 0
        return X.apply(lambda x: 1 if x == 'spam' else 0)
