import unittest
import os
import shutil

import pandas as pd

from . import test_utils
from merlin import MERLIN
from merlin.data import TabularDataManager

dir_path = os.path.dirname(os.path.realpath(__file__))
(X_left, predicted_labels_left,
 X_right, predicted_labels_right) = test_utils.setup_synthetic_tabular()


class TestMERLINTabular(unittest.TestCase):

    def test_0_data_manager(self):
        '''
        '''
        if os.path.exists(f'{dir_path}/test_results'):
            shutil.rmtree(f'{dir_path}/test_results')
        os.mkdir(f'{dir_path}/test_results')

        self.assertEqual(len(X_left), len(predicted_labels_left),
                         'Different prediction and X sizes in t1')
        self.assertEqual(len(X_right), len(predicted_labels_right),
                         'Different prediction and X sizes in t2')

        TabularDataManager(X_left, predicted_labels_left, 'left')
        TabularDataManager(X_right, predicted_labels_right, 'right')

    def test_1_trace_instantiation(self):
        '''
        '''
        MERLIN(
            X_left, predicted_labels_left,
            X_right, predicted_labels_right,
            data_type='tabular', surrogate_type='sklearn',
            hyperparameters_selection=False,
            save_path=f'{dir_path}/test_results',
            save_surrogates=False, save_bdds=False
        )

    def test_2_run_trace(self):
        '''
        '''
        exp = MERLIN(
            X_left, predicted_labels_left,
            X_right, predicted_labels_right,
            data_type='tabular', surrogate_type='sklearn',
            hyperparameters_selection=False,
            save_path=f'{dir_path}/test_results',
            save_surrogates=False, save_bdds=False
        )

        percent_mc = exp.trace.run_montecarlo(threshold=0.5)
        self.assertEqual(percent_mc, 0.2, f'Percent mc is {percent_mc}')
        exp.run_trace(1)

        n_classes = len(pd.read_csv(f'{dir_path}/test_results/trace.csv',
                        sep=';')['class_id'].unique())
        self.assertEqual(n_classes, 2,
                         f'Error: expected 2 classes in csv, found {n_classes}.')

    def test_3_explain(self):
        '''
        '''
        exp = MERLIN(
            X_left, predicted_labels_left,
            X_right, predicted_labels_right,
            data_type='tabular', surrogate_type='sklearn',
            hyperparameters_selection=False,
            save_path=f'{dir_path}/test_results',
            save_surrogates=False, save_bdds=False
        )
        exp.run_trace()
        exp.run_explain()

    def test_4_rulefit(self):
        '''
        '''
        exp = MERLIN(
            X_left, predicted_labels_left,
            X_right, predicted_labels_right,
            data_type='tabular', surrogate_type='rulefit',
            hyperparameters_selection=False,
            save_path=f'{dir_path}/test_results',
            save_surrogates=False, save_bdds=False
        )
        exp.run_trace()
        exp.run_explain()


if __name__ == '__main__':
    unittest.main()
