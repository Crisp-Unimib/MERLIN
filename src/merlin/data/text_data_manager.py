import logging
import ast
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import resample
from merlin.util.helpers import text_formatter
from merlin.data.data_manager import DataManager


class TextDataManager(DataManager):
    '''
    '''

    def __init__(self, X, Y_predicted, time_label):
        '''
        '''
        X = pd.Series([self.check_var_names(x) for x in X])
        super().__init__(X, Y_predicted, time_label)
        self.sparse_samples = None
        self.df = pd.DataFrame({
            'X': self.X.copy(),
            'Y_predicted': self.Y_predicted.copy(),
        })

    def generate_data_predictions(self, percent_dataset: int):
        '''Generated predictions for both datasets.

        Parameters
        ----------
        percent_dataset : int
            Describes the percentage of the dataset to use.
        '''
        self.logger.info(
            f'Sampling dataset with percent: {percent_dataset} and saving labels...')
        self.logger.info(f'Total dataset n is {self.df.shape[0]}.')

        if percent_dataset < 1:
            data = self.df.sample(frac=percent_dataset,
                                  replace=False, random_state=42).sort_index()
        else:
            data = self.df.copy()
        self.logger.info(f'N. Samples {self.time_label}: {data.shape[0]}')
        self.logger.info('Training vectorizer...')

        onehot_vectorizer = CountVectorizer(binary=True, lowercase=False)

        onehot_vectorizer.fit(data['X'])
        self.sparse_samples = onehot_vectorizer.transform(data['X'])
        self.feature_names = onehot_vectorizer.get_feature_names()

        for i, class_id in enumerate(self.classes):

            data_class = data.copy()
            class_id = str(class_id)
            # Balancing
            n_positive_class = sum(data_class['Y_predicted'] == class_id)
            n_negative_class = sum(data_class['Y_predicted'] != class_id)
            try:
                data_class = pd.concat([
                    # data_class[data_class['Y_predicted'] == class_id],
                    resample(data_class[data_class['Y_predicted'] == class_id],
                             replace=True,     # sample with replacement
                             n_samples=n_negative_class,    # to match majority class
                             random_state=42),  # reproducible results,
                    data_class[data_class['Y_predicted'] != class_id]
                    # data_class[data_class['Y_predicted'] != class_id].sample(
                    #     n=n_positive_class, replace=False, random_state=42)
                ])
            except ValueError:
                pass
            # data_class.to_csv(f'results/{self.time_label}_{class_id}.csv', index=False)

            self.surrogate_train_data[class_id] = onehot_vectorizer.transform(
                data_class['X'])

            if data_class['Y_predicted'].dtype == np.dtype(int) or data_class['Y_predicted'].dtype == np.dtype(np.int64):
                self.Y_predicted_binarized[class_id] = np.array(
                    [1 if int(x) == i else 0 for x in data_class['Y_predicted']])
            else:
                self.Y_predicted_binarized[class_id] = np.array(
                    [1 if x == class_id else 0 for x in data_class['Y_predicted']])

        self.logger.info(f'Finished predicting {self.time_label}')

    def get_rule_occurrence(self, rule):
        '''Returns a number of occurrences in the corpus for a specific rule.
        '''
        rule_dict = ast.literal_eval(re.sub(r'(\w+):', r'"\1":', rule))
        col_ids = [self.feature_names.index(
            item) for item in rule_dict.keys() if item in self.feature_names]
        dense_data = pd.DataFrame(self.sparse_samples[:, col_ids].todense())
        rule_values = pd.Series(rule_dict.values())
        return (dense_data == rule_values).all(axis=1)

    def count_rule_occurrence(self, rule):
        rule_matches = self.get_rule_occurrence(rule)
        return sum(rule_matches)

    def get_rule_examples(self, rule, class_id, typ='add', n_examples=5):
        '''Get n examples in the dataset for a given rule.
        '''
        rule_matches = self.get_rule_occurrence(rule)
        rule_dict = ast.literal_eval(re.sub(r'(\w+):', r'"\1":', rule))
        true_rules_set = {x[0] for x in rule_dict.items() if x[1] == 1}
        category_matches = self.Y_predicted == str(class_id)
        matches = rule_matches & category_matches
        tot_n = sum(matches)

        try:
            perc = round(tot_n * 100 / sum(rule_matches), 2)
        except ZeroDivisionError:
            perc = 0
        print(f'Rule {typ}: {rule}')
        print(
            f'Overall, the rule appeared {sum(rule_matches)} times in {self.time_label}.')
        if tot_n == 0:
            return None, tot_n, perc
        print(
            f'Out of these, {tot_n} ({perc}%) belong to the class {class_id}.')
        print('Some example instances:')

        padding = 4
        examples = self.X[matches].drop_duplicates().head(n_examples).values
        for example in examples:

            indexes = []
            example = example.split(' ')
            for word in true_rules_set:
                if (example.index(word) != -1):
                    indexes.append(example.index(word))
            if len(indexes) == 0:
                print('Error: nothing found')
            else:
                st = '- '
                for index in indexes:
                    if index > padding:
                        st += f'[...] {" ".join(example[index - padding:index])} '
                    else:
                        st += f'{" ".join(example[:index])} '
                    if typ == 'del':
                        st += text_formatter(example[index],
                                             bc=1, tc=16, bold=True)
                    if typ == 'add':
                        st += text_formatter(example[index],
                                             bc=2, tc=16, bold=True)
                    if typ == 'still':
                        st += text_formatter(example[index],
                                             bc=3, tc=16, bold=True)
                    if index + padding < len(example):
                        st += f' {" ".join(example[index+1:index+1+padding])} [...]'
                    else:
                        st += f' {" ".join(example[index+1:])}'
                st = st.replace('[...][...]', '[...]')
                print(st)
        return examples, tot_n, perc
