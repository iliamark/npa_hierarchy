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
    