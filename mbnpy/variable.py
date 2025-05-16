import numpy as np
from itertools import chain, combinations
from mbnpy.extern.tabulate import tabulate

class Variable:
    """
    A class to manage information about a variable used in matrix-based Bayesian networks.

    Attributes
    ----------
    name : str
        Name of the variable.
    values : list
        Description of basic states.
    B : list of set
        Mapping from basic states to composite states. Each element is a set
        representing a group of basic states (automatically generated).

    Notes
    -----
    **How to read the mapping matrix:**

    Each row in the matrix represents a composite state.
    Each column corresponds to a basic state.

    For example, with 3 basic states, the mapping matrix B is:

        [[1, 0, 0],
         [0, 1, 0],
         [0, 0, 1],
         [1, 1, 0],
         [1, 0, 1],
         [0, 1, 1],
         [1, 1, 1]]

    - The first row `[1, 0, 0]` means that State 0 includes only State 0.
    - The sixth row `[0, 1, 1]` means that State 5 includes States 1 and 2.
    - The last row `[1, 1, 1]` means that State 6 includes all basic states.

    **Best practice when assigning `values`:**

    When applicable, use an ordering where lower indices represent worse outcomes, as some modules assume this ordering.
    For example: `['failure', 'survival']` since `0 < 1`.
    """

    def __init__(self, name: str, values: list=[], B_flag: str='auto'):
        '''Initialise the Variable object.

        Args:
            name (str): name of the variable.
            values (list or np.ndarray): description of states.
            B_flag (str): flag to determine how B is generated.
        '''
        assert isinstance(name, str), 'name should be a string'
        assert isinstance(values, (list, np.ndarray)), \
            'values must be a list or np.ndarray'
        assert B_flag in ['auto', 'store', 'fly'], \
            'B_flag must be either auto, store, or fly'

        self._name = name
        self._values = values
        self._B_flag = B_flag

        if len(self.values) > 0:
            if B_flag == 'fly':
                self._B = None
            elif B_flag == 'store':
                self._B = self.gen_B()
            elif B_flag == 'auto':
                if len(self.values) <= 6:
                    self._B = self.gen_B()
                else:
                    self._B = None
        else:
            self._B = None

    # Magic methods
    def __hash__(self):
        return hash(self._name)

    def __lt__(self, other):
        return self._name < other.name

    def __eq__(self, other):
        if isinstance(other, Variable):
            return self._name == other._name and self._values == other._values
        else:
            return False

    def __repr__(self):
        #if self._B is None:
        #    return (
        #        f"<Variable representing {repr(self._name)} at {hex(id(self))},\n"
        #        f"values={repr(self._values)},\n"
        #        f"B_flag={repr(self._B_flag)})"
        #    )
        #else:
        #    b = [(i, f'{x}') for i, x in enumerate(self._B)]
        return (
                f"<Variable representing {repr(self._name)}({repr(self._values)}) at {hex(id(self))}>\n"
                #f"values={repr(self._values)}\n"
                #f"{tabulate(b, headers=['idx', 'value'], tablefmt='grid')}"
            )

    def __str__(self):
        if self._B is None:
            return (
                #f"<Variable representing {repr(self._name)} at {hex(id(self))}>\n"
                f"<Variable representing {repr(self._name)}({repr(self._values)}) at {hex(id(self))}>\n"
                #f"values={repr(self._values)}\n"
                f"B_flag={repr(self._B_flag)})"
            )
        else:
            header = ['idx', 'B']
            b = [(i, f'{x}') for i, x in enumerate(self._B)]
            return (
                #f"<Variable representing {repr(self._name)} at {hex(id(self))}>\n"
                f"<Variable representing {repr(self._name)}({repr(self._values)}) at {hex(id(self))}>\n"
                #f"values={repr(self._values)}\n"
                f"{tabulate(b, headers=header, tablefmt='grid')}"
            )

    # Property for 'name'
    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_name):
        assert isinstance(new_name, str), 'name must be a string'
        self._name = new_name

    # Property for 'values'
    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, new_values):
        assert isinstance(new_values, list), 'values must be a list'
        self._values = new_values
        self.update_B()

    # Property for 'B_flag'
    @property
    def B_flag(self):
        return self._B_flag

    @B_flag.setter
    def B_flag(self, new_B_flag):
        assert new_B_flag in ['auto', 'store', 'fly'], \
            'B_flag must be either auto, store, or fly'
        self._B_flag = new_B_flag
        self.update_B()

    # Method to generate the mapping matrix
    def gen_B(self):
        n = len(self._values)
        B = [
            set(x)
            for x in chain.from_iterable(
                combinations(range(n), r) for r in range(1, n+1)
            )
        ]
        return B

    # Method to update the mapping matrix
    def update_B(self):
        if len(self._values) > 0:
            if self._B_flag == 'store':
                self._B = self.gen_B()
            elif self._B_flag == 'fly':
                self._B = None
            elif self._B_flag == 'auto':
                if len(self._values) <= 6:
                    self._B = self.gen_B()
                else:
                    self._B = None
        else:
            self._B = None

    # Property for 'B'
    @property
    def B(self):
        return self._B


    def get_state(self, state_set):
        '''Finds the state index of a given set of basic states.

        The sets are ordered as follows (cf. gen_B):
        [{0}, {1}, ..., {n-1}, {0, 1}, {0, 2}, ..., {n-2, n-1},
        {0, 1, 2}, ..., {0, 1, ..., n-1}]

        Args:
            state_set (set): set of basic states.

        Returns:
            state (int): state index in B matrix of the given set.
        '''
        assert isinstance(state_set, set), 'set must be a set'

        if self.B is not None:
            # Find the index directly from B
            state = self.B.index(state_set)
        else:
            # Find the index by calculation
            # The number of elements in the target set
            num_elements = len(state_set)
            # Number of basic states
            n = len(self.values)

            # Initialize the state
            state = 0
            # Add the number of sets with fewer elements
            for k in range(1, num_elements):
                state += len(list(combinations(range(n), k)))
            # Find where the target set is in the group
            # with 'num_elements' elements
            combinations_list = list(combinations(range(n), num_elements))

            # Convert target_set to a sorted tuple
            # to match the combinations
            target_tuple = tuple(sorted(state_set))
            # Find the index within the group
            idx_in_group = combinations_list.index(target_tuple)

            # Add the position within the group to the state
            state += idx_in_group

        return state

    def get_set(self, state):
        '''Finds the set of basic states represented by a given state index.

        The sets are ordered as follows (cf. gen_B):
        [{0}, {1}, ..., {n-1}, {0, 1}, {0, 2}, ..., {n-2, n-1},
        {0, 1, 2}, ..., {0, 1, ..., n-1}]

        Args:
            state (int): state index.

        Returns:
            set (set): set of basic states.
        '''
        assert np.issubdtype(type(state), np.integer), 'state must be an integer'

        if self.B is not None:
            return self.B[state]
        else:
            # the number of states
            n = len(self.values)
            # Initialize the state tracker
            current_state = 0

            # Iterate through the group sizes
            # (1-element sets, 2-element sets, etc.)
            for k in range(1, n+1):
                # Count the number of sets of size k
                comb_count = len(list(combinations(range(n), k)))

                # Check if the index falls within this group
                if current_state + comb_count > state:
                    # If it falls within this group,
                    # calculate the exact set
                    combinations_list = list(combinations(range(n), k))
                    set_tuple = combinations_list[state - current_state]
                    return set(set_tuple)

                # Otherwise, move to the next group
                current_state += comb_count

            # If the index is out of bounds, raise an error
            raise IndexError(f"The given state index must be not greater than {2**n-1}")


    def get_state_from_vector(self, vector):
        '''Finds the state index for a given binary vector.

        Args:
            vector (list or np.ndarray): binary vector.
            1 if the basic state is involved, 0 otherwise.

        Returns:
            state (int): state index.
            -1 if the vector is all zeros.
        '''
        assert isinstance(vector, (list, np.ndarray)), \
            'vector must be a list or np.ndarray'
        assert len(vector) == len(self.values), \
            'vector must have the same length as values'

        # Count the number of 1's in the vector
        num_ones = sum(vector)

        # Return -1 if the vector is all zeros
        if num_ones == 0:
            return -1

        # Number of basic states
        n = len(vector)

        # Initialize the state
        state = 0
        # Add the number of vectors with fewer 1's
        for k in range(1, num_ones):
            state += len(list(combinations(range(n), k)))

        # Find where this vector is in the group with 'num_ones' ones
        one_positions = [i for i, val in enumerate(vector) if val == 1]
        # Find the position of this specific combination in the group
        combs = list(combinations(range(n), num_ones))
        idx_in_group = combs.index(tuple(one_positions))

        # Add the position within the group to the state
        state += idx_in_group

        return state

    def get_Bst_from_Bvec( self, Bvec ):
        '''Converts a binary vector into its corresponding state index.

        Args:
           Bvec (np.ndarray): (x*y*z) binary array.
           x is the number of instances for the first Cpm object.
           y is the number of instances for the second Cpm object.
           z is the number of basic states.

        Returns:
            Bst (np.ndarray): (x*y) integer array.
            Each element represents the state index of
            the corresponding binary vector in Bvec.
        '''
        Bst = np.apply_along_axis(self.get_state_from_vector, -1, Bvec)
        return Bst

