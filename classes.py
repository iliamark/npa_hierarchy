import numpy as np
import itertools
import warnings
import cvxpy as cp

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
        

class NpaHierarchy:
    """
    Represents a hierarchy of NPA (Navascués-Pironio-Acin) operators.
    Args:
        bell_scenario (tuple): A tuple of 4 positive integers representing the Bell scenario (NA, NB, NX, NM).
        npa_depth (int): The depth of the NPA hierarchy.
    Raises:
        ValueError: If bell_scenario is not a tuple of 4 positive integers or npa_depth is not a positive integer.
    Attributes:
        bell_scenario (tuple): A tuple of 4 positive integers representing the Bell scenario (NA, NB, NX, NM).
        npa_depth (int): The depth of the NPA hierarchy.
        A (ndarray): An array of integers from 0 to NA-1.
        B (ndarray): An array of integers from 0 to NB-1.
        X (ndarray): An array of integers from 0 to NX-1.
        M (ndarray): An array of integers from 0 to NM-1.
        npa_operators (list): A list of NpaOperator objects generated based on the specified bell scenario and npa depth.
        gamma_operator_matrix (list): A matrix of NpaOperator objects representing the gamma operators.
    Methods:
        npa_operator_generator(): Generates a list of NpaOperator objects based on the specified bell scenario and npa depth.
        _remove_duplicates(lst): Removes duplicates from a list.
        npa_gamma_matrix_generator(): Generates the gamma operator matrix.
        feasability(distribution): Computes the gamma matrix instance and variable dictionary based on a given distribution.
    """
    def __init__(self, bell_scenario, npa_depth):
        if not isinstance(bell_scenario, tuple) or len(bell_scenario) != 4 or not all(isinstance(n, int) and n > 0 for n in bell_scenario):
            raise ValueError("bell_scenario must be a tuple of 4 positive integers")
        if not isinstance(npa_depth, int) or npa_depth <= 0:
            raise ValueError("npa_depth must be a positive integer")
        self.bell_scenario = bell_scenario
        NA,NB,NX,NM = self.bell_scenario
        self.npa_depth = npa_depth
        self.A = np.array(range(NA))
        self.B = np.array(range(NB))
        self.X = np.array(range(NX))
        self.M = np.array(range(NM))
        self.npa_operators = self.npa_operator_generator()
        self.gamma_operator_matrix = self.npa_gamma_matrix_generator()

    def npa_operator_generator(self):
        """
        Generates a list of NpaOperator objects based on the specified bell scenario and npa depth.

        Returns:
            list: A list of NpaOperator objects.

        """
        npa_entries = list()
        def recursive_npa_entry(cur_depth, cur_entry, entry_list):
            if cur_depth == 0:
                return entry_list.append(cur_entry)
            else:
                for x in self.X:
                    for a in self.A:
                        new_entry = cur_entry @ NpaOperator(Operator((a,x)), Operator.identity())
                        recursive_npa_entry(cur_depth - 1, new_entry, entry_list)
                for m in self.M:
                    for b in self.B:
                        new_entry = cur_entry @ NpaOperator(Operator.identity(), Operator((b,m)))
                        recursive_npa_entry(cur_depth - 1, new_entry, entry_list)
        for i in range(self.npa_depth + 1):
            recursive_npa_entry(i, NpaOperator(Operator.identity(), Operator.identity()), npa_entries)
        return NpaHierarchy._remove_duplicates(npa_entries) 
    
    @staticmethod
    def _remove_duplicates(lst):
        new_lst = []
        for item in lst:
            if item not in new_lst:
                new_lst.append(item)
        return new_lst
    
    def npa_gamma_matrix_generator(self):
        npa_operator_count = len(self.npa_operators)
        gamma_matrix_entries = list()
        for i in range(npa_operator_count):
            gamma_row = list()
            for j in range(npa_operator_count):
                gamma_row.append(self.npa_operators[i].conj() @ self.npa_operators[j])
            gamma_matrix_entries.append(gamma_row)
        return gamma_matrix_entries
    
    def feasability(self, distribution):
        # todo - i want to do an upper triangle, diagonal and then using the symmetricity.
        gamma_matrix_instance = list()
        variable_dict = dict()
        for i in range(len(self.gamma_operator_matrix)):
            gamma_matrix_instance_row = list()
            for j in range(len(self.gamma_operator_matrix[i])):
                current_operator = self.gamma_operator_matrix[i][j]
                if current_operator.is_variable():
                    if current_operator.__str__() in variable_dict:
                        current_value = variable_dict[current_operator.__str__()]
                    elif current_operator.conj().__str__() in variable_dict:
                        current_value = variable_dict[current_operator.conj().__str__()]
                    else:
                        current_value = cp.Variable()
                        variable_dict[current_operator.__str__()] = current_value
                else:
                    current_value = current_operator.evaluate(distribution)
                gamma_matrix_instance_row.append(current_value)
            gamma_matrix_instance.append(gamma_matrix_instance_row)
        return gamma_matrix_instance, variable_dict       
    
    def feasability_symm(self, distribution):
        # todo - i want to do an upper triangle, diagonal and then using the symmetricity.
        n = len(self.gamma_operator_matrix)
        gamma_matrix_instance = [[0 for i in range(n)] for j in range(n)]
        variable_dict = dict()
        for i in range(n):
            j = 0
            while j <= i:
                current_operator = self.gamma_operator_matrix[i][j]
                if current_operator.is_variable():
                    if current_operator.__str__() in variable_dict:
                        current_value = variable_dict[current_operator.__str__()]
                    elif current_operator.conj().__str__() in variable_dict:
                        current_value = variable_dict[current_operator.conj().__str__()]
                    else:
                        current_value = cp.Variable()
                        variable_dict[current_operator.__str__()] = current_value
                else:
                    current_value = current_operator.evaluate(distribution)
            gamma_matrix_instance[i][j] = current_value
            if j < i:
                gamma_matrix_instance[j][i] = current_value
        return gamma_matrix_instance, variable_dict       