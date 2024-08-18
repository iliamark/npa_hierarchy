import numpy as np
import itertools
import warnings

int_types = (int, np.int32, np.int64, np.integer)

class Distribution:
    # default constructor
    def __init__(self, vals, bell_scenario):
        # Normalization check
        self.bell_scenario = bell_scenario
        NA,NB,NX,NM = bell_scenario
        self.A = np.array(range(NA))
        self.X = np.array(range(NX))
        self.B = np.array(range(NB))
        self.M = np.array(range(NM))
        self.vals = np.copy(vals)
        if self.normalization_check() == False:
            warnings.warn("This is a warning message -- The distribution is not normalized.", UserWarning)

    def __call__(self, a, b, x, m):
        return self.vals[a,b,x,m]
    
    def __add__(self, other):
        vals = self.vals + other.vals
        bell_scenario = self.bell_scenario
        return Distribution(vals, bell_scenario)
    
    def __mul__(self, other):
        if isinstance(other, int_types + (float,)):
            return Distribution(other*self.vals, self.bell_scenario)
        
    def __rmul__(self, other):
        if isinstance(other, int_types + (float,)):
            return self.__mul__(other)
    
    def normalization_check(self):
        for x in self.X:
            for m in self.M:
                sum = 0
                for a in self.A:
                    for b in self.B:
                        sum += self(a,b,x,m)
                if sum != 1:
                    return False
        return True

    def singnaling_check(self):
        for a, x, m0, m1 in itertools.product(self.A, self.X, self.M, self.M):
            p_a_xm0 = 0
            p_a_xm1 = 0
            for b in self.B:
                p_a_xm0 += self(a,b,x,m0)
                p_a_xm1 += self(a,b,x,m1)
            if p_a_xm0 != p_a_xm1:
                return True
        
        for b, m, x0, x1 in itertools.product(self.B, self.M, self.X, self.X):
            p_b_x0m = 0
            p_b_x1m = 0
            for a in self.A:
                p_b_x0m += self(a,b,x0,m)
                p_b_x1m += self(a,b,x1,m)
            if p_b_x0m != p_b_x1m:
                return True
        return False
    
    def marginal(self, output, input, party):
        p_input = 0.5
        sum = 0
        if party == 0:
            x = input
            a = output
            for b in self.B:
                for m in self.M:
                    sum += p_input * self(a,b,x,m)
        elif party == 1:
            m = input
            b = output
            for a in self.A:
                for x in self.X:
                    sum += p_input * self(a,b,x,m)
        else:
            raise Exception("Party not supported!")
        return sum
    
    def chsh_score(self):
        score = 0
        for a, b, x, m in itertools.product(self.A, self.B, self.X, self.M):
            xor = bool(a) ^ bool(b)
            prod = x*m
            if xor == prod:
                score += self(a,b,x,m)
        score /= 4
        return score

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
        if self == NpaOperator.null():
            return False
        elif self == NpaOperator.identity():
            return False
        elif self.right_operator == Operator.identity() and len(self.left_operator.transcript) == 1:
            return False
        elif self.left_operator == Operator.identity() and len(self.right_operator.transcript) == 1:
            return False
        elif len(self.left_operator.transcript) == 1 and len(self.right_operator.transcript) == 1:
            return False
        else:
            return True