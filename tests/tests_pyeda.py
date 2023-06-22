import unittest
from itertools import combinations
from pyeda.boolalg.expr import And, Or, Xor, Not
from pyeda.inter import expr, expr2bdd


class TestPyEda(unittest.TestCase):

    def test_expr(self):
        '''
        '''
        def get_constraint_v1(n_variables):
            var_list = [f'Var{n+1}' for n in range(n_variables)]
            uneven_number_is_true = Xor(*var_list)
            max_one_is_true = []
            for var1, var2 in combinations(var_list, 2):
                max_one_is_true.append(Not(And(var1, var2)))
            max_one_is_true = And(*max_one_is_true)
            only_one_is_true = And(uneven_number_is_true, max_one_is_true)

            expr_constraint = expr(only_one_is_true)
            return list(expr_constraint.satisfy_all())

        def get_constraint_v2(n_variables):
            var_list = [f'Var{n+1}' for n in range(n_variables)]
            constraint = []
            for i in range(n_variables):
                true_var = var_list[i]
                false_vars = [Not(var) for var in var_list[:i]+var_list[i+1:]]
                constraint.append(And(true_var, *false_vars))
            constraint = Or(*constraint)

            expr_constraint = expr(constraint)
            return list(expr_constraint.satisfy_all())


        for n_variables in range(10):
            satisfy_list =  get_constraint_v1(n_variables)
            self.assertEqual(len(satisfy_list), n_variables,
                             f'Number of satisfy v1 is different than starting n ({n_variables})')
            self.assertEqual(sum([sum(x.values()) for x in satisfy_list]), n_variables,
                             f'Sum of values in satisfy v1 list is different than starting n ({n_variables})')
            satisfy_list =  get_constraint_v2(n_variables)
            self.assertEqual(len(satisfy_list), n_variables,
                             f'Number of satisfy v2 is different than starting n ({n_variables})')
            self.assertEqual(sum([sum(x.values()) for x in satisfy_list]), n_variables,
                             f'Sum of values in satisfy v2 list is different than starting n ({n_variables})')


if __name__ == '__main__':
    unittest.main()
