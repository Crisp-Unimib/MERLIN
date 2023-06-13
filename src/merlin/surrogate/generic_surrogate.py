import logging
from merlin.util.logger import build_logger
from sklearn.model_selection import train_test_split


class GenericSurrogate:
    '''
    Base surrogate object
    '''

    def __init__(self, X, predicted_labels, time_label,
                 class_id, feature_names, test_size=0.2):
        '''
        '''
        self.logger = build_logger(
            logging.INFO, __name__, 'logs/surrogate.log')
        
        if test_size and test_size > 0:
            (
                self.X,
                self.X_score,
                self.predicted_labels,
                self.predicted_labels_score
            ) = train_test_split(
                X,
                predicted_labels,
                test_size=test_size,
                random_state=42
            )
        else:
            self.X = X
            self.X_score = X
            self.predicted_labels = predicted_labels
            self.predicted_labels_score = predicted_labels
        self.time_label = time_label
        self.class_id = class_id
        self.feature_names = feature_names

    def hyperparameters_selection(self):
        pass

    def fit(self):
        pass

    def score(self):
        pass

    def surrogate_to_bdd_string(self):
        pass

    def save_surrogate_img(self, save_path):
        pass
