import numpy as np
import itertools
import warnings
import cvxpy as cp
from Operator import Operator

int_types = (int, np.int32, np.int64, np.integer)



class NpaOperator:
    VARIABLE_ENTRY = -1
    def __init__(self, left_operator, right_operator):
        if isinstance(left_operator, Operator) and isinstance(right_operator, Operator):
            self.left_operator = left_operator
            self.right_operator = right_operator
        else:
            raise ValueError("Invalid input")
    
    @staticmethod
    def identity():
        return NpaOperator(Operator.identity(), Operator.identity())

    @staticmethod
    def null():
        return NpaOperator(Operator.null(), Operator.null())

    def __matmul__(self, other):
        new_left = self.left_operator @ other.left_operator
        new_right = self.right_operator @ other.right_operator
        if new_left == Operator.null() or new_right == Operator.null():
            return NpaOperator.null()
        else:
            return NpaOperator(new_left, new_right)
        
    def __eq__(self, other):
        if not isinstance(other, NpaOperator):
            raise ValueError("Invalid input to __eq__")
        return self.left_operator == other.left_operator and self.right_operator == other.right_operator
    
    def conj(self):
        return NpaOperator(self.left_operator.conj(), self.right_operator.conj())
    
    def __str__(self):
        left_str = self.left_operator.__str__()
        right_str = self.right_operator.__str__()
        return f"NpaOp[{left_str}, {right_str}]"
    
    def __repr__(self): return self.__str__()

    def evaluate(self, distribution):
        if self == NpaOperator.null():
            return 0
        elif self == NpaOperator.identity():
            return 1
        elif self.right_operator == Operator.identity() and len(self.left_operator.transcript) == 1:
            a,x = self.left_operator.transcript[0]
            return distribution.marginal(a, x, 0)
        elif self.left_operator == Operator.identity() and len(self.right_operator.transcript) == 1:
            b,m = self.right_operator.transcript[0]
            return distribution.marginal(b, m, 1)
        elif len(self.left_operator.transcript) == 1 and len(self.right_operator.transcript) == 1:
            a,x = self.left_operator.transcript[0]
            b,m = self.right_operator.transcript[0]
            return distribution(a,b,x,m)
        else:
            raise ValueError("Invalid operator for manifestation")
        
    def is_variable(self):
        # If there is more than one projection in either the left or right operator, then the operator is not a variable.
        not_var_cond = len(self.left_operator.transcript) == 1 and len(self.right_operator.transcript) == 1
        return not not_var_cond
            