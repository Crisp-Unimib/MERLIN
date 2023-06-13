import logging
import ast
import re
from merlin.util.logger import build_logger
from merlin.data.data_manager import DataManager

import numpy as np
import pandas as pd

from sklearn import preprocessing


class TabularDataManager(DataManager):
    '''
    '''

    def __init__(self, X, Y_predicted, time_label):
        '''
        '''
        super().__init__(X, Y_predicted, time_label)
        self.X = X.reset_index(drop=True)
        self.X.columns = [self.check_column_names(x) for x in X.columns]
        self.df = pd.DataFrame(self.X.copy())
        self.df['Y_predicted'] = self.Y_predicted.copy()

    def generate_data_predictions(self, percent_dataset: int):
        '''Generated predictions for both datasets.

        Parameters
        ----------
        percent_dataset : int
            Describes the percentage of the dataset to use.

        '''
        self.logger.info(
            f'Sampling dataset with percent: {percent_dataset} and saving labels...')

        data = self.df.sample(frac=percent_dataset,
                              replace=False, random_state=42).sort_index()
        self.logger.info(f'N. Samples {self.time_label}: {data.shape[0]}')

        self.feature_names = list(self.X.columns)

        max_n = data['Y_predicted'].value_counts().min()

        for i, class_id in enumerate(self.classes):

            data_class = data.copy()
            class_id = str(class_id)
            # Balancing
            data_class = pd.concat([
                data_class[data_class['Y_predicted'] == class_id].sample(
                    n=max_n, replace=False, random_state=42),
                data_class[data_class['Y_predicted'] != class_id].sample(
                    n=max_n, replace=False, random_state=42),
            ]).reset_index(drop=True)
            # data_class.to_csv(f'results/{self.time_label}_{class_id}.csv', index=False)

            assert sum(data_class['Y_predicted'] == class_id) == sum(
                data_class['Y_predicted'] != class_id)

            self.surrogate_train_data[class_id] = data_class[self.feature_names].copy(
            )

            if data_class['Y_predicted'].dtype == np.dtype(int) or data_class['Y_predicted'].dtype == np.dtype(np.int64):
                self.Y_predicted_binarized[class_id] = np.array(
                    [1 if int(x) == i else 0 for x in data_class['Y_predicted']])
            else:
                self.Y_predicted_binarized[class_id] = np.array(
                    [1 if x == class_id else 0 for x in data_class['Y_predicted']])

        self.logger.info(f'Finished predicting {self.time_label}')

    def get_rule_occurrence(self, rule):
        '''Returns a number of occurrences in the dataset for a specific rule.
        '''
        rule_dict = ast.literal_eval(re.sub(r'(\w+):', r'"\1":', rule))
        condition = True
        for key, direction in rule_dict.items():
            if key.split('LEQ')[0] not in self.X.columns:
                continue
            if 'LEQ' in key:
                var_name, quantile = key.split('LEQ')
                quantile = int(quantile)
                if direction == 1:
                    item = (self.X[var_name].astype('int') <= quantile)
                else:
                    item = (self.X[var_name].astype('int') > quantile)
            else:
                item = (self.X[key].astype('int') == direction)
            condition = condition & item
        return condition

    def count_rule_occurrence(self, rule):
        rule_matches = self.get_rule_occurrence(rule)
        return sum(rule_matches)

    def get_rule_examples(self, rule, class_id, n_examples=5):
        rule_matches = self.get_rule_occurrence(rule)
        category_matches = self.Y_predicted == str(class_id)
        matches = rule_matches & category_matches
        tot_n = sum(matches)

        try:
            perc = round(tot_n * 100 / sum(rule_matches), 2)
        except ZeroDivisionError:
            perc = 0
        print(f'Rule: {rule}')
        print(
            f'Overall, the rule appeared {sum(rule_matches)} times in {self.time_label}.')
        if tot_n == 0:
            return None, tot_n, perc
        print(
            f'Out of these, {tot_n} ({perc}%) belong to the class {class_id}.')
        # print('Some example instances:')
        examples = self.X[matches].drop_duplicates().head(n_examples)
        return examples, tot_n, perc
