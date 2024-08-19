import numpy as np
import itertools
import warnings
import cvxpy as cp
from NpaOperator import NpaOperator
from Operator import Operator

int_types = (int, np.int32, np.int64, np.integer)

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
            recursive_npa_entry(i, NpaOperator.identity(), npa_entries)
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
    
    def compute_gamma_matrix(self, distribution):
        n = len(self.gamma_operator_matrix)
        gamma_matrix_instance = [[0 for _ in range(n)] for _ in range(n)]
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
                    # The matrix entry is not a variable. Evaluate the operator based on the distribution. 
                    current_value = current_operator.evaluate(distribution)
                gamma_matrix_instance[i][j] = current_value
                if i < j:
                    # The matrix is symmetric. Fill the upper right triangle.
                    gamma_matrix_instance[j][i] = current_value
                j += 1
        return gamma_matrix_instance, variable_dict       
