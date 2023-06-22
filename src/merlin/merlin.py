import logging

from .trace import Trace
from .explain import Explain

from merlin.data import TextDataManager
from merlin.data import TabularDataManager


class MERLIN():
    '''
    '''

    def __init__(self,
                 X_left, predicted_labels_left,
                 X_right, predicted_labels_right,
                 data_type='text',
                 hyperparameters_selection=False,
                 hyperparameters={},
                 log_level=logging.INFO,
                 save_path='results',
                 graphviz_path='C:/Program Files (x86)/Graphviz2.38/bin',
                 surrogate_type='sklearn',
                 surrogate_test_size=0.2,
                 save_surrogates=False,
                 save_csvs=True,
                 save_bdds=False):

        self.log_level = log_level
        self.hyperparameters_selection = hyperparameters_selection
        self.hyperparameters = hyperparameters
        self.save_path = save_path
        self.graphviz_path = graphviz_path
        self.surrogate_type = surrogate_type
        self.surrogate_test_size = surrogate_test_size
        self.save_surrogates = save_surrogates
        self.save_csvs = save_csvs
        self.save_bdds = save_bdds
        self.trace_results = None
        self.explain_results = None

        unsupported_combinations = [
            ('text', 'rulefit')
        ]

        if (data_type, surrogate_type) in unsupported_combinations:
            raise NotImplementedError(f'Surrogate {surrogate_type} is '
                                      f'not supported with {data_type} data.')

        if data_type == 'text':
            self.data_manager = {
                'left': TextDataManager(X_left, predicted_labels_left, 'left'),
                'right': TextDataManager(X_right, predicted_labels_right, 'right'),
            }
        else:
            self.data_manager = {
                'left': TabularDataManager(X_left, predicted_labels_left, 'left'),
                'right': TabularDataManager(X_right, predicted_labels_right, 'right'),
            }

        self.trace = Trace(
            self.data_manager,
            log_level=self.log_level,
            hyperparameters_selection=self.hyperparameters_selection,
            hyperparameters=self.hyperparameters,
            save_path=self.save_path,
            surrogate_type=self.surrogate_type,
            surrogate_test_size=self.surrogate_test_size,
            save_surrogates=self.save_surrogates,
            save_csvs=self.save_csvs,
        )
        self.explain = None

    def run_trace(self, percent_dataset: float = 1):
        '''
        '''
        self.trace.run_trace(percent_dataset)
        self.trace_results = self.trace.results

    def run_explain(self):
        '''
        '''
        self.explain = Explain(
            self.data_manager,
            save_path=self.save_path,
            graphviz_path=self.graphviz_path,
            trace_results=self.trace_results,
            log_level=self.log_level,
            save_bdds=self.save_bdds,
            save_csvs=self.save_csvs,
        )
        self.explain.run_explain()
        self.explain_results = self.explain.results