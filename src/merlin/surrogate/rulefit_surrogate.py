from time import time
import re
from typing import Dict
# from pydot import graph_from_dot_data
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor
from rulefit import RuleFit
from merlin.surrogate import GenericSurrogate


class RulefitSurrogate(GenericSurrogate):
    '''
    '''

    def __init__(self, X, predicted_labels, time_label, class_id,
                 feature_names, hyperparameters, test_size=0.2):
        '''
        '''
        super().__init__(X, predicted_labels, time_label,
                         class_id, feature_names, test_size)

        self.X = self.X.values.astype('float')
        self.X_score = self.X_score.values.astype('float')
        self.hyperparameters = hyperparameters
        self.bdd = None
        self.paths = {}
        self.fit_time = None
        self.surrogate_predictions = None
        self.fidelity = None
        gb = GradientBoostingRegressor(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.01,
            random_state=42,
        )
        self._model = RuleFit(tree_generator=gb, random_state=42)

    def fit(self):
        '''Fit the model.
        '''
        start_time = time()

        self._model.fit(
            self.X,
            np.array(self.predicted_labels),
            feature_names=np.array(self.feature_names),
        )
        self.score()

        self.fit_time = round(time() - start_time, 3)
        self.logger.info(f'Time for fitting surrogate: {self.fit_time}')

    def score(self):
        '''Compute fidelity score.
        '''

        pred = self._model.predict(self.X_score)
        # converting the predictions to 0 and 1
        self.surrogate_predictions = np.array(
            [1 if x >= 0.5 else 0 for x in pred])

        self.fidelity = {
            'f1_binary': metrics.f1_score(self.predicted_labels_score,
                                          self.surrogate_predictions,
                                          average='binary'),
            'f1_macro': metrics.f1_score(self.predicted_labels_score,
                                         self.surrogate_predictions,
                                         average='macro'),
            'f1_weighted': metrics.f1_score(self.predicted_labels_score,
                                            self.surrogate_predictions,
                                            average='weighted'),
            'recall_binary': metrics.recall_score(self.predicted_labels_score,
                                                  self.surrogate_predictions,
                                                  average='binary'),
            'recall_weighted': metrics.recall_score(self.predicted_labels_score,
                                                    self.surrogate_predictions,
                                                    average='weighted'),
            'precision_binary': metrics.precision_score(self.predicted_labels_score,
                                                        self.surrogate_predictions,
                                                        average='binary'),
            'precision_weighted': metrics.precision_score(self.predicted_labels_score,
                                                          self.surrogate_predictions,
                                                          average='weighted'),
            'balanced_accuracy': metrics.balanced_accuracy_score(self.predicted_labels_score,
                                                                 self.surrogate_predictions),
        }
        self.fidelity = {k: round(v, 3) for k, v in self.fidelity.items()}

        self.logger.debug(self.predicted_labels_score[:100])
        self.logger.debug(self.surrogate_predictions[:100])
        self.logger.info(f'Fidelity of the surrogate: {self.fidelity}')
        self.logger.info(metrics.classification_report(
            self.predicted_labels_score, self.surrogate_predictions))

    def surrogate_to_bdd_string(self):
        '''Transform RuleFit surrogate to BDD string.
        '''
        def prune(rule_df):
            """
            Prunes the rules to get up to threshold % of importance
            """
            rule_df = rule_df.sort_values(
                "importance", ascending=False).reset_index(drop=True)
            sum_importance = rule_df['importance'].sum()
            rule_df['importance_percent'] = rule_df['importance'] / \
                sum_importance
            rule_df['cumsum'] = rule_df['importance_percent'].cumsum()
            res = rule_df[rule_df['cumsum'] <
                          self.hyperparameters['importance_threshold']]
            return res

        def formatter(text):
            """
            Transforms rulefit rules texts to the correct format
            e.g:
            charge_clinic <= 0.5 & fortunately <= 0.5 --> ~charge_clinic & ~fortunately
            """
            features = text.split('&')
            res = ''
            for feature in features:
                feature_spl = re.split('[> < <= >=]', feature)
                feature_spl = [x for x in feature_spl if x != '']

                if float(feature_spl[1]) == 0.5:
                    # Binary case
                    if '<' in feature:
                        res += f' {feature_spl[0]} &'
                    elif '>' in feature:
                        res += f' ~{feature_spl[0]} &'
                else:
                    # Ordinal case
                    if '<' in feature:
                        res += f' {feature_spl[0]}LEQ{str(int(float(feature_spl[1]) - 0.5))} &'
                    elif '>' in feature:
                        res += f' ~{feature_spl[0]}LEQ{str(int(float(feature_spl[1]) - 0.5))} &'
            return res.rstrip('&').strip()

        def create_path(rule_df):
            """
            concatenate rules with '|'
            """
            df = prune(rule_df)
            rules = df['rule']
            res = [formatter(rule) for rule in rules]
            return ' | '.join(res)

        self.logger.info('Transforming surrogate to BDD string...')

        self.bdd = []
        rules = self._model.get_rules()
        rules = rules[(rules.coef != 0) & (rules['type'] == 'rule')]
        df_rule = pd.DataFrame(rules)
        self.bdd = create_path(df_rule)

        def save_paths_n(rule_df):
            for _, row in rule_df.iterrows():
                self.paths[formatter(row['rule'])
                           ] = row['support'] * int(self.X.shape[0])

        save_paths_n(df_rule)
        self.logger.debug(f'BDD String for class {self.class_id}: {self.bdd}')
