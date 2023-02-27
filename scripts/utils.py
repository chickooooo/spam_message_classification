import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

import scripts.custom_pipelines as cp


class Setup:
    """Loads the model and required dependencies. Prepares the model for making predictions
    """

    def __init__(self) -> None:
        # load model
        self.__model = joblib.load("./models/logistic_regression.pkl")

        # create text cleaner
        cleaner = cp.TextCleaner()
        # load tfidf vectorizer
        tfidf_vect = joblib.load("./models/tfidf_vectorizer.pkl")

        # create full pipeline
        self.__full_pipeline = Pipeline([
            ('text_cleaner', cleaner),
            ('tfidf_vect', tfidf_vect),
        ])

    def predict(self, X: list[str]) -> list[str]:
        # convert to series
        x_series = pd.Series(X)

        # process input data
        x_processed = self.__full_pipeline.transform(x_series)

        # make predictions
        predictions = self.__model.predict(x_processed)
        
        # return labeled the predictions
        return ['spam' if value == 1 else 'ham' for value in predictions]
