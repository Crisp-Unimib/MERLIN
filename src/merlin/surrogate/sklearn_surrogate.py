from time import time
import os
import random
from typing import Dict
from pydot import graph_from_dot_data

import numpy as np

from sklearn import tree
from sklearn import metrics
from sklearn.tree._tree import TREE_UNDEFINED, TREE_LEAF
from sklearn.model_selection import RandomizedSearchCV
from merlin.surrogate import GenericSurrogate


class SklearnSurrogate(GenericSurrogate):
    '''
    '''

    def __init__(self, X, predicted_labels, time_label, class_id,
                 feature_names, hyperparameters, test_size=0.2):
        '''
        '''
        super().__init__(X, predicted_labels, time_label,
                         class_id, feature_names, test_size)

        self.hyperparameters = hyperparameters
        self.bdd = None
        self.paths = {}
        self.fit_time = None
        self.surrogate_predictions = None
        self.fidelity = None
        self._model = tree.DecisionTreeClassifier(
            splitter='best',
            random_state=42,
            class_weight='balanced',
            **self.hyperparameters
        )

    def hyperparameters_selection(self, param_grid: Dict = None, cv: int = 5):
        '''
        '''
        start_time = time()
        np.random.seed(42)
        random.seed(42)
        self.logger.info('Beginning hyperparameters selection...')
        default_grid = {
            'criterion': ['gini', 'entropy'],
            # restrict the minimum number of samples in a leaf
            'min_samples_leaf': [0.01, 0.02],
            'max_depth': [3, 5, 7],  # helps in reducing the depth of the tree
            # restrict the minimum % of samples before splitting
            'min_samples_split': [0.01, 0.02, 0.03],
        }
        search_space = param_grid if param_grid is not None else default_grid
        # Cost function aiming to optimize(Total Cost) = measure of fit + measure of complexity
        # References for pruning:
        # 1. http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        # 2. https://www.coursera.org/lecture/ml-classification/optional-pruning-decision-trees-to-avoid-overfitting-qvf6v
        # Using Randomize Search here to prune the trees to improve readability without
        # comprising on model's performance
        verbose_level = 4 if self.logger.level >= 20 else 0
        random_search_estimator = RandomizedSearchCV(estimator=self._model, cv=cv,
                                                     param_distributions=search_space,
                                                     scoring='f1', n_iter=10, n_jobs=-1,
                                                     random_state=42, verbose=verbose_level)
        # train a surrogate DT
        random_search_estimator.fit(self.X, self.predicted_labels)
        # access the best estimator
        self._model = random_search_estimator.best_estimator_

        self.hyperparameters['max_depth'] = self._model.max_depth
        self.hyperparameters['min_samples_split'] = self._model.min_samples_split

        self.logger.info(
            f'Time for fitting surrogate: {round(time() - start_time, 3)}')
        self.logger.info(f'Best model: {self._model}')

    def fit(self):
        '''Fit the model.
        '''
        start_time = time()

        np.random.seed(42)
        random.seed(42)
        self._model.fit(self.X, self.predicted_labels)

        def is_leaf(inner_tree, index):
            # Check whether node is leaf node
            return (inner_tree.children_left[index] == TREE_LEAF and
                    inner_tree.children_right[index] == TREE_LEAF)

        def prune_index(inner_tree, decisions, index=0):
            # Start pruning from the bottom - if we start from the top, we might miss
            # nodes that become leaves during pruning.
            # Do not use this directly - use prune_duplicate_leaves instead.
            if not is_leaf(inner_tree, inner_tree.children_left[index]):
                prune_index(inner_tree, decisions,
                            inner_tree.children_left[index])
            if not is_leaf(inner_tree, inner_tree.children_right[index]):
                prune_index(inner_tree, decisions,
                            inner_tree.children_right[index])

            # Prune children if both children are leaves now and make the same decision:
            if (is_leaf(inner_tree, inner_tree.children_left[index]) and
                is_leaf(inner_tree, inner_tree.children_right[index]) and
                (decisions[index] == decisions[inner_tree.children_left[index]]) and
                    (decisions[index] == decisions[inner_tree.children_right[index]])):
                # turn node into a leaf by "unlinking" its children
                inner_tree.children_left[index] = TREE_LEAF
                inner_tree.children_right[index] = TREE_LEAF
                inner_tree.feature[index] = TREE_UNDEFINED
                # print("Pruned {}".format(index))

        def prune_duplicate_leaves(mdl):
            # Remove leaves if both
            decisions = mdl.tree_.value.argmax(
                axis=2).flatten().tolist()  # Decision for each node
            prune_index(mdl.tree_, decisions)

        prune_duplicate_leaves(self._model)

        self.score()

        self.fit_time = round(time() - start_time, 3)
        self.logger.info(f'Time for fitting surrogate: {self.fit_time}')

    def score(self):
        '''Compute fidelity score.
        '''
        self.surrogate_predictions = self._model.predict(self.X_score)

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

        self.logger.debug(self.predicted_labels_score[:20])
        self.logger.debug(self.surrogate_predictions[:20])
        self.logger.info(f'Fidelity of the surrogate: {self.fidelity}')
        self.logger.info(metrics.classification_report(self.predicted_labels_score,
                                                       self.surrogate_predictions))

        self.logger.info('Fidelity on training:')
        self.surrogate_predictions_training = self._model.predict(self.X)
        self.logger.info(metrics.classification_report(self.predicted_labels,
                                                       self.surrogate_predictions_training))

    def surrogate_to_bdd_string(self):
        '''Transform surrogate tree to BDD string using depth first search.
        '''
        self.logger.info('Transforming surrogate to BDD string...')
        stack = []
        self.bdd = []

        def _tree_recurse(node):
            if self._model.tree_.feature[node] == TREE_UNDEFINED:
                # Leaf node, base case
                value = np.argmax(self._model.tree_.value[node][0])
                if value == 1:
                    path = ' & '.join(stack[:])
                    self.bdd.append(path)
                    self.paths[path] = self._model.tree_.n_node_samples[node]
                return

            # Recursion case
            name = self.feature_names[self._model.tree_.feature[node]]
            if self._model.tree_.threshold[node] != 0.5:
                # Case where feature is LEQ variable
                name = f'{name}LEQ{str(self._model.tree_.threshold[node])}'
            stack.append(f'~{name}')
            self.logger.debug(stack)

            _tree_recurse(self._model.tree_.children_left[node])

            stack.pop()
            self.logger.debug(stack)
            stack.append(name)
            self.logger.debug(stack)

            _tree_recurse(self._model.tree_.children_right[node])

            stack.pop()
            self.logger.debug(stack)

        _tree_recurse(0)
        self.bdd = ' | '.join(self.bdd)
        self.logger.info(f'BDD String for class {self.class_id}: {self.bdd}')

    def save_surrogate_img(self, save_path):
        '''Save decision tree surrogates to image.
        '''
        directory = f'{save_path}/surrogate_tree'
        if not os.path.exists(directory):
            os.makedirs(directory)
        fname = f'{directory}/{self.class_id}_{self.time_label}.png'
        graph_str = tree.export_graphviz(
            self._model,
            class_names=[f'NOT {self.class_id}', self.class_id],
            feature_names=self.feature_names,
            filled=True
        )
        (graph,) = graph_from_dot_data(graph_str)
        self.logger.info(f'Saving {fname} to disk')
        graph.write_png(fname)
