import numpy as np
import itertools
import warnings

int_types = (int, np.int32, np.int64, np.integer)


class Operator:
    IDENTITY_TRANSCRIPT = ((1,),)
    NULL_TRANSCRIPT = ((0,),)
    def __init__(self, input):
        if isinstance(input, Operator):
            self.transcript = input.transcript
        elif isinstance(input, tuple) and (len(input) == 2 and (all(isinstance(n, int_types) for n in input))):
            self.transcript = (input,)
        else:
            raise ValueError("Unexpected input to operator constructor")
    
    @staticmethod
    def identity():
        id_operator = Operator((-1,-1))
        id_operator.transcript = Operator.IDENTITY_TRANSCRIPT
        return id_operator
    
    @staticmethod
    def null():
        null_operator = Operator((-1,-1))
        null_operator.transcript = Operator.NULL_TRANSCRIPT
        return null_operator

    def __eq__(self, other):
        if isinstance(other, Operator):
            return self.transcript == other.transcript
        else:
            raise ValueError("Unexpected input to __eq__ method")

    def __matmul__(self, other):
        if isinstance(other, Operator):
            if self == Operator.null() or other == Operator.null():
                return Operator.null()
            elif self == Operator.identity():
                return Operator(other)
            elif other == Operator.identity():
                return Operator(self)
            else:
                suffix = self.transcript[-1]
                prefix = other.transcript[0]
                # Same operator
                if suffix == prefix:
                    new_op = Operator.identity()
                    new_op.transcript = self.transcript[:-1] + other.transcript
                    return new_op
                # Same measurement, different outcome
                elif suffix[-1] == prefix[-1]:
                    return Operator.null()
                # Different measurements altogether
                else:
                    new_op = Operator.identity()
                    new_op.transcript = self.transcript + other.transcript
                    return new_op
        else:
            raise ValueError("Only operators instances allowed with @ operation.")
        
    def conj(self):
        new_op = Operator.identity()
        new_op.transcript = self.transcript[::-1]
        return new_op
    
    def __str__(self):
        return f"Op{self.transcript}"