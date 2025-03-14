import numpy as np
import textwrap
import copy
import warnings
from scipy.stats import norm, beta

from mbnpy.variable import Variable
from mbnpy import utils


class Cpm(object):
    '''Defines the conditional probability matrix (CPM).
    CPM plays the same role as CPT in conventional Bayesian networks.
    Ref: Byun et al. (2019). Matrix-based Bayesian Network for
    efficient memory storage and flexible inference. 
    Reliability Engineering & System Safety, 185, 533-545.

    Attributes:
        variables (list): list of instances of Variable objects.
        no_child (int): number of child nodes.
        C (array_like): event matrix.
        p (array_like): probability vector related to C.
        Cs (array_like): event matrix of samples.
        q (array_like): sampling probability vector related to Cs.
        sample_idx (array_like): sample index vector related to Cs.
        ps (array_like): probability vector of samples.
        Sample weights are calculated as ps/q.

    Notes:
        C and p have the same number of rows.
        Cs, q, sample_idx, and ps have the same number of rows.
    '''

    def __init__(self, variables, no_child, C=[], p=[], Cs=[], q=[], ps=[], sample_idx=[]):

        self.variables = variables
        self.no_child = no_child
        self.C = C
        self.p = p
        self.Cs = Cs
        self.q = q
        self.ps = ps
        self.sample_idx = sample_idx

    # Magic methods
    def __hash__(self):
        return hash(self._name)
    
    def __eq__(self, other):
        if isinstance(other, Cpm):
            return (
                self._name == other._name and
                self._variables == other._variables and
                self._no_child == other._no_child and
                self._C == other._C and
                self._p == other._p and
                self._Cs == other._Cs and
                self._q == other._q and
                (
                    (not self._ps or not other._ps or self._ps == other._ps) and
                    (not self._sample_idx or not other._sample_idx or 
                     self._sample_idx == other._sample_idx)
                )
            )
        else:
            return False

    def __repr__(self):
        details = [
            f"{self.__class__.__name__}(",
            f"    variables={self.get_names()},",
            f"    no_child={self.no_child},",
            f"    C={self.C},",
            f"    p={self.p},",
        ]

        if self._Cs:
            details.append(f"    Cs={self._Cs},")
        if self._q:
            details.append(f"    q={self._q},")
        if self._ps:
            details.append(f"    ps={self._ps},")
        if self._sample_idx:
            details.append(f"    sample_idx={self._sample_idx},")

        details.append(")")
        return "\n".join(details)
    
    # Properties
    @property
    def variables(self):
        return self._variables

    @variables.setter
    def variables(self, value):
        assert isinstance(value, list), 'variables must be a list of Variable'
        assert all([isinstance(x, Variable) for x in value]), 'variables must be a list of Variable'
        self._variables = value

    @property
    def no_child(self):
        return self._no_child

    @no_child.setter
    def no_child(self, value):
        assert isinstance(value, (int, np.int32, np.int64)), 'no_child must be a numeric scalar'
        assert value <= len(self.variables), f'no_child must be less than or equal to the number of variables: {value}, {len(self.variables)}'
        self._no_child = value

    @property
    def C(self):
        return self._C

    @C.setter
    def C(self, value):
        if isinstance(value, list):
            value = np.array(value, dtype=np.int32)

        if value.size:
            assert value.dtype in (np.dtype('int64'), np.dtype('int32')), f'Event matrix C must be a numeric matrix: {value}'

            if value.ndim == 1:
                value.shape = (len(value), 1)
            else:
                assert value.shape[1] == len(self._variables), 'C must have the same number of columns as that of variables'

            max_C = np.max(value, axis=0, initial=0)
            max_var = [2**len(x.values)-1 for x in self._variables]
            assert all(max_C <= max_var), f'check C matrix: {max_C} vs {max_var}'

        self._C = value

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, value):
        if isinstance(value, list):
            value = np.array(value)[:, np.newaxis]

        if value.ndim == 1:
            value.shape = (len(value), 1)

        if value.size:
            assert len(value) == self._C.shape[0], 'p must have the same length as the number of rows in C'

            all(isinstance(y, (float, np.float32, np.float64, int, np.int32, np.int64)) for y in value), 'p must be a numeric vector'

        self._p = value

    @property
    def Cs(self):
        return self._Cs

    @Cs.setter
    def Cs(self, value):
        if isinstance(value, list):
            value = np.array(value, dtype=np.int32)

        if value.size:
            if value.ndim == 1: # event matrix for samples
                value.shape = (len(value), 1)
            else:
                assert value.shape[1] == len(self._variables), 'Cs must have the same number of columns as the number of variables'

            max_Cs = np.max(value, axis=0, initial=0)
            max_var_basic = [len(x.values) for x in self.variables]
            assert all(max_Cs <= max_var_basic), f'check Cs matrix: {max_Cs} vs {max_var_basic}'

        self._Cs = value

    @property
    def q(self):
        return self._q

    @q.setter
    def q(self, value):
        if isinstance(value, list):
            value = np.array(value)[:, np.newaxis]

        if value.ndim == 1:
            value.shape = (len(value), 1)

        if value.size and self._Cs.size:
            assert len(value) == self._Cs.shape[0], 'q must have the same length as the number of rows in Cs'

        if value.size:
            all(isinstance(y, (float, np.float32, np.float64, int, np.int32, np.int64)) for y in value), 'q must be a numeric vector'

        self._q = value

    @property
    def ps(self):
        return self._ps

    @ps.setter
    def ps(self, value):
        if isinstance(value, list):
            value = np.array(value)[:, np.newaxis]

        if value.ndim == 1:
            value.shape = (len(value), 1)

        if value.size and self._Cs.size:
            assert len(value) == self._Cs.shape[0], 'ps must have the same length as the number of rows in Cs'

            all(isinstance(y, (float, np.float32, np.float64, int, np.int32, np.int64)) for y in value), 'p must be a numeric vector'

        self._ps = value

    @property
    def sample_idx(self):
        return self._sample_idx

    @sample_idx.setter
    def sample_idx(self, value):
        if isinstance(value, list):
            value = np.array(value)[:, np.newaxis]

        if value.ndim == 1:
            value.shape = (len(value), 1)

        if value.size and self._Cs.size:
            assert len(value) == self._Cs.shape[0], 'sample_idx must have the same length as the number of rows in Cs'

            all(isinstance(y, (float, np.float32, np.float64, int, np.int32, np.int64)) for y in value), 'p must be a numeric vector'

        self._sample_idx = value    


    def product(self, M):
        """
        Returns an instance of Cpm
        M: instance of Cpm
        var: a dict of instances of Variable
        """

        assert isinstance(M, Cpm), f'M should be an instance of Cpm'

        ch_names1 = [x.name for x in self.variables[:self.no_child]]
        ch_names2 = [x.name for x in M.variables[:M.no_child]]
        check = set(ch_names1).intersection(ch_names2)
        assert not bool(check), f'PMFs must not have common child nodes: {ch_names1}, {ch_names2}'

        if self.C.size or self.Cs.size:
            idx_vars, _ = ismember(self.variables, M.variables)
            prod_vars = M.variables + get_value_given_condn(self.variables, flip(idx_vars))

            new_child = self.variables[:self.no_child] + M.variables[:M.no_child]

            new_parent = self.variables[self.no_child:] + M.variables[M.no_child:]
            new_parent, _ = setdiff(new_parent, new_child)

            if new_parent:
                new_vars = new_child + new_parent
            else:
                new_vars = new_child
            _, idx_vars2 = ismember(new_vars, prod_vars)

            Mprod = Cpm(variables=new_vars,
                        no_child = len(new_child))
        else:
            Mprod = M

        # Product starts here
        if self._C.size:
            if M._C.size:
                Cnew, pnew = C_prod_C(self, M, new_vars)
                Mprod.C, Mprod.p = Cnew, pnew

                if M._Cs.size and self._Cs.size:
                    Cs_new, q_new, ps_new, sample_idx_new = Cs_prod_Cs(
                    self.Cs, M.Cs, self.variables, M.variables,
                    self.q, M.q, self.ps, M.ps, self.sample_idx, M.sample_idx, new_vars)

                    Mprod.Cs, Mprod.q = Cs_new, q_new
                    Mprod.ps, Mprod.sample_idx = ps_new, sample_idx_new

                elif M._Cs.size and not self._Cs.size:
                    Cs_new, q_new, ps_new, sample_idx_new = C_prod_Cs(self, M, new_vars)
                    Mprod.Cs, Mprod.q = Cs_new, q_new
                    Mprod.ps, Mprod.sample_idx = ps_new, sample_idx_new

                elif self._Cs.size and not M._Cs.size:
                    Cs_new, q_new, ps_new, sample_idx_new = C_prod_Cs(M, self, new_vars)
                    Mprod.Cs, Mprod.q = Cs_new, q_new
                    Mprod.ps, Mprod.sample_idx = ps_new, sample_idx_new
            
            elif M._Cs.size and not self._Cs.size:
                Cs_new, q_new, ps_new, sample_idx_new = C_prod_Cs(self, M, new_vars)
                Mprod.Cs, Mprod.q = Cs_new, q_new
                Mprod.ps, Mprod.sample_idx = ps_new, sample_idx_new

            elif self._Cs.size:
                Cs_new1, q_new1, ps_new1, sample_idx_new1 = C_prod_Cs(self, M, new_vars)
                Cs_new2, q_new2, ps_new2, sample_idx_new2 = Cs_prod_Cs(
                    self.Cs, M.Cs, self.variables, M.variables,
                    self.q, M.q, self.ps, M.ps, self.sample_idx, M.sample_idx, new_vars)

                Cs_new = np.vstack([Cs_new1, Cs_new2])
                q_new = np.concatenate([q_new1, q_new2])
                ps_new = np.concatenate([ps_new1, ps_new2])
                sample_idx_new = np.concatenate([sample_idx_new1.flatten(), sample_idx_new2.flatten()])

                # Sort all sample arrays based on sample_idx
                sort_idx = np.argsort(sample_idx_new)
                Cs_new = Cs_new[sort_idx]
                q_new = q_new[sort_idx]
                ps_new = ps_new[sort_idx]
                sample_idx_new = sample_idx_new[sort_idx]

                Mprod.Cs, Mprod.q, = Cs_new.astype(int), q_new
                Mprod.ps, Mprod.sample_idx = ps_new, sample_idx_new.astype(int)

        else:
            if M._C.size and M._Cs.size:
                Cs_new1, q_new1, ps_new1, sample_idx_new1 = C_prod_Cs(M, self, new_vars)
                Cs_new2, q_new2, ps_new2, sample_idx_new2 = Cs_prod_Cs(
                    M.Cs, self.Cs, M.variables, self.variables,
                    M.q, self.q, M.ps, self.ps, M.sample_idx, self.sample_idx, new_vars)

                Cs_new = np.vstack([Cs_new1, Cs_new2])
                q_new = np.vstack([q_new1, q_new2])
                ps_new = np.concatenate([ps_new1, ps_new2])
                sample_idx_new = np.concatenate([sample_idx_new1, sample_idx_new2])

                sort_idx = np.argsort(sample_idx_new.flatten())
                Cs_new = Cs_new[sort_idx]
                q_new = q_new[sort_idx]
                ps_new = ps_new[sort_idx]

                Mprod.Cs, Mprod.q = Cs_new, q_new
                Mprod.ps, Mprod.sample_idx = ps_new, sample_idx_new

            elif M._C.size:
                Cs_new, q_new, ps_new, sample_idx_new = C_prod_Cs(M, self, new_vars)
                Mprod.Cs, Mprod.q = Cs_new, q_new
                Mprod.ps, Mprod.sample_idx = ps_new, sample_idx_new

            elif M._Cs.size:
                Cs_new, q_new, ps_new, sample_idx_new = Cs_prod_Cs(
                    self.Cs, M.Cs, self.variables, M.variables,
                    self.q, M.q, self.ps, M.ps, self.sample_idx, M.sample_idx, new_vars)

                Mprod.Cs, Mprod.q, = Cs_new, q_new
                Mprod.ps, Mprod.sample_idx = ps_new, sample_idx_new

        if Mprod.Cs.size:
            Mprod.Cs = Mprod.Cs.astype(int)

        return  Mprod
    
    def _get_Cnew(self, M, new_vars):
        """
        For internal use in "product" method
        """

        new_vars_tf1, new_vars_idx1 = ismember( new_vars, self.variables )
        new_vars_tf2, new_vars_idx2 = ismember( new_vars, M.variables )

        n_row1 = len(self.C)
        n_row2 = len(M.C)

        Bst_dict = {}
        for k, v in enumerate(new_vars):

            if new_vars_tf1[k] and new_vars_tf2[k]: # common variable to product

                n_val = len(v.values)

                Bvec1 = np.zeros((n_row1, n_val), dtype=int)
                Bvec2 = np.zeros((n_row2, n_val), dtype=int)

                C1 = self.C[:, new_vars_idx1[k]]
                for i, b in enumerate(C1):
                    st = list( v.B[ b ] )
                    Bvec1[i, st] = 1
                Bvec1 = np.tile(Bvec1, (n_row2, 1, 1))

                C2 = M.C[:, new_vars_idx2[k]]
                for i, b in enumerate(C2):
                    st = list( v.B[ b ] )
                    Bvec2[i, st] = 1
                Bvec2 = np.tile(Bvec2[:, np.newaxis, :], (1, n_row1, 1))

                Bvec = Bvec1 * Bvec2
                Bst = v.get_Bst_from_Bvec( Bvec )
                
                Bst_dict[v.name] = Bst

            elif new_vars_tf1[k]: # it is in self only
                C1 = self.C[:, new_vars_idx1[k]]
                Bst = np.tile(C1.T, (n_row2, 1, 1))
                Bst = np.squeeze(Bst)

                Bst_dict[v.name] = Bst

            else: # it is in M only
                C2 = M.C[:, new_vars_idx2[k]]
                Bst = np.repeat(C2.reshape(-1,1), n_row1, axis=1)

                Bst_dict[v.name] = Bst

        Cnew_list = []
        for v in new_vars:
            Cnew_list.append( Bst_dict[v.name].T.reshape(-1) )

        # Cnew with -1
        # -1 elements mean incompatible rows between the two CPMs
        Cnew_with_minus1 = np.column_stack( Cnew_list ) 

        return Cnew_with_minus1
    
    def _C_prod_C(self, M, new_vars):
        """
        For internal use in "product" method
        """

        new_vars_tf1, new_vars_idx1 = ismember( new_vars, self.variables )
        new_vars_tf2, new_vars_idx2 = ismember( new_vars, M.variables )

        n_row1 = len(self.C)
        n_row2 = len(M.C)

        Bst_dict = {}
        for k, v in enumerate(new_vars):

            if new_vars_tf1[k] and new_vars_tf2[k]: # common variable to product

                n_val = len(v.values)

                Bvec1 = np.zeros((n_row1, n_val), dtype=int)
                Bvec2 = np.zeros((n_row2, n_val), dtype=int)

                C1 = self.C[:, new_vars_idx1[k]]
                for i, b in enumerate(C1):
                    st = list( v.B[ b ] )
                    Bvec1[i, st] = 1
                Bvec1 = np.tile(Bvec1, (n_row2, 1, 1))

                C2 = M.C[:, new_vars_idx2[k]]
                for i, b in enumerate(C2):
                    st = list( v.B[ b ] )
                    Bvec2[i, st] = 1
                Bvec2 = np.tile(Bvec2[:, np.newaxis, :], (1, n_row1, 1))

                Bvec = Bvec1 * Bvec2
                Bst = v.get_Bst_from_Bvec( Bvec )
                
                Bst_dict[v.name] = Bst

            elif new_vars_tf1[k]: # it is in self only
                C1 = self.C[:, new_vars_idx1[k]]
                Bst = np.tile(C1.T, (n_row2, 1, 1))
                Bst = np.squeeze(Bst)

                Bst_dict[v.name] = Bst

            else: # it is in M only
                C2 = M.C[:, new_vars_idx2[k]]
                Bst = np.repeat(C2.reshape(-1,1), n_row1, axis=1)

                Bst_dict[v.name] = Bst

        Cnew_list = []
        for v in new_vars:
            Cnew_list.append( Bst_dict[v.name].T.reshape(-1) )
        Cnew = np.column_stack( Cnew_list )

        pnew = np.repeat(self.p, n_row2) * np.tile(M.p, (n_row1, 1)).flatten()

        mask = np.sum( Cnew < 0, axis=1 ) < 1

        Cnew = Cnew[mask]
        pnew = pnew[mask]

        return Cnew, pnew

    def get_variables(self, item):

        if isinstance(item, str):
            return [x for x in self.variables if x.name == item][0]
        elif isinstance(item, list):
            return [self.get_variables(y) for y in item]


    def get_names(self):
        return [x.name for x in self.variables]


    def get_subset(self, row_idx, flag=True, isC=True):
        """
        Returns the subset of Cpm

        Parameters
        ----------
        row_idx: array like
        flag: boolean
            default True, 0 if exclude row_idx
        isC: boolean
            if True, C and p are reduced; if False, Cs, q, ps, sample_idx are.
        """

        assert flag in (0, 1)
        assert isC in (0, 1)

        if isC:
            if flag:
                assert set(row_idx).issubset(range(self.C.shape[0]))
            else:
                # select row excluding the row_index
                row_idx, _ = setdiff(range(self.C.shape[0]), row_idx)

            if self.p.size:
                p_sub = self.p[row_idx]
            else:
                p_sub = []

            Mnew = Cpm(variables=self.variables,
                       no_child=self.no_child,
                       C=self.C[row_idx,:],
                       p=p_sub,
                       Cs = self.Cs,
                       q = self.q,
                       ps = self.ps,
                       sample_idx = self.sample_idx)

        else:
            if flag:
                assert set(row_idx).issubset(range(self.Cs.shape[0]))
            else:
                # select row excluding the row_index
                row_idx, _ = setdiff(range(self.Cs.shape[0]), row_idx)

            if self.q.size:
                q_sub = self.q[row_idx]
            else:
                q_sub = []

            if self.ps.size:
                ps_sub = self.ps[row_idx]
            else:
                ps_sub = []

            if self.sample_idx.size:
                si_sub = self.sample_idx[row_idx]
            else:
                si_sub = []

            Mnew = Cpm(variables=self.variables,
                       no_child=self.no_child,
                       Cs=self.Cs[row_idx,:],
                       q=q_sub,
                       ps=ps_sub,
                       sample_idx = si_sub)

        return Mnew


    def get_means(self, names):
        """
        Get means of variables in names
        INPUT:
        names: a list of names
        OUTPUT:
        means: a list of means (the same order as names)
        """
        assert isinstance(names, list), 'names should be a list'
        assert len(set(names))==len(names), f'names has duplicates: {names}'

        idx = [self.get_names().index(x) for x in names]

        return [(self.C[:, i]*self.p[:, 0]).sum() for i in idx]


    def iscompatible(self, M, composite_state=True):
        """
        Returns a boolean list (n,)

        Parameters
        ----------
        M: instance of Cpm for compatibility check
        composite_state: False if the same rows are returned;
            True if composite states and basic state can be
            considered compatible if they are.
        """

        assert isinstance(M, Cpm), f'M should be an instance of Cpm'

        assert ( (M.C.shape[0] == 1) and (not M.Cs.size) ) or ( (M.Cs.shape[0] == 1) and (not M.C.size) ), 'C/Cs must be a single row'

        if self.C.size and M.C.size:
            is_cmp = iscompatible(self.C, self.variables, M.variables, M.C[0], composite_state)

        else:
            is_cmp = []

        return is_cmp


    def get_col_ind(self, names):
        """
        INPUT:
        names: a list of variable names
        OUTPUT:
        idx: a list of column indices of v_names
        """

        assert isinstance(names, list), f'names should be a list'

        assert len(set(names)) == len(names), f'names has duplicates: {names}'

        return [self.get_names().index(x) for x in names]
    
    def get_col(self, names):

        assert isinstance(names, list), f'names should be a list'

        assert len(set(names)) == len(names), f'names has duplicates: {names}'

        inds = [self.get_names().index(x) for x in names]

        C = np.zeros((len(self.C), len(names)), dtype=int)
        for i, col_ind in enumerate(inds):
            C[:,i] = self.C[:,col_ind]

        return C



    def merge(self, M):

        assert isinstance(M, Cpm), f'M should be an instance of Cpm'
        assert self.variables == M.variables, 'must have the same scope and order'

        new_cpm = copy.copy(self)

        cs = self.C.tolist()

        for cx, px in zip(M.C.tolist(), M.p.tolist()):
            try:
                new_cpm.p[cs.index(cx)] += px
            except IndexError:
                new_cpm.C = np.vstack((new_cpm.C, cx))
                new_cpm.p = np.vstack((new_cpm.p, px))

        new_cpm.Cs = np.vstack((self.Cs, M.Cs))
        new_cpm.q = np.vstack((self.q, M.q))
        new_cpm.ps = np.vstack((self.ps, M.ps))
        new_cpm.sample_idx = np.vstack((self.sample_idx, M.sample_idx))

        return new_cpm


    def sum(self, variables, flag=True):
        """
        Returns instance of Cpm with based on Sum over CPMs.

        Parameters
        ----------
        variables: list of variables or names of variables
        flag: boolean
            1 (default) - sum out variables, 0 - leave only variables
        """

        assert isinstance(variables, list), 'variables should be a list'
        if variables and isinstance(variables[0], str):
            variables = self.get_variables(variables)

        if flag and set(self.variables[self.no_child:]).intersection(variables):
            print('Parent nodes are NOT summed up')

        if flag:
            vars_rem, vars_rem_idx = setdiff(self.variables[:self.no_child], variables)

        else:
            # FIXME
            _, vars_rem_idx = ismember(variables, self.variables[:self.no_child])
            vars_rem_idx = get_value_given_condn(vars_rem_idx, vars_rem_idx)
            vars_rem_idx = sorted(vars_rem_idx)
            vars_rem = [self.variables[x] for x in vars_rem_idx]

        no_child_sum = len(vars_rem)

        if self.variables[self.no_child:]:
            vars_rem += self.variables[self.no_child:]
            vars_rem_idx += list(range(self.no_child, len(self.variables)))

        _variables = [self.variables[i] for i in vars_rem_idx]

        M = Cpm(variables=_variables,
                C=self.C[:, vars_rem_idx],
                no_child=len(vars_rem_idx),
                p = self.p)

        Csum = np.array([], dtype=int).reshape(0, len(vars_rem))
        psum = []

        while M.C.size:

            Mc = M.get_subset([0])
            is_cmp = iscompatible(M.C, vars_rem, vars_rem, Mc.C, composite_state=False)

            Csum = np.vstack((Csum, Mc.C[0]))

            if M.p.size:
                psum.append(np.sum(M.p[is_cmp]))

            M = M.get_subset(np.where(is_cmp)[0], flag=0)

        Ms = Cpm(variables=vars_rem,
                 no_child=no_child_sum,
                 C=Csum,
                 p=np.reshape(psum, (-1, 1)))

        if self.Cs.size:

            Cs = self.Cs[:, vars_rem_idx].copy()

            Ms.Cs = Cs.astype(int)
            Ms.q = self.q.copy()
            Ms.ps = self.ps.copy()
            Ms.sample_idx = self.sample_idx.copy()

        return Ms


    def get_prob(self, var_inds, var_states, flag=True):

        assert isinstance(var_inds, list), 'var_inds should be a list'

        if var_inds and isinstance(var_inds[0], str):
            var_inds = self.get_variables(var_inds)

        assert isinstance(var_states, (list, np.ndarray)), 'var_states should be an array'

        assert len(var_inds) == len(var_states), f'"var_inds" {var_inds} and "var_states" {var_states} must have the same length.'

        assert flag in (0, 1), 'Operation flag must be either 1 (keeping given row indices default) or 0 (deleting given indices)'

        _var_states = []
        for i, x in enumerate(var_states):
            if isinstance(x, str):
                assert x in var_inds[i].values, f'{x} not in {var_inds[i].values}'
                _var_states.append(var_inds[i].values.index(x))

        if _var_states:
            var_states = _var_states[:]

        if isinstance(var_states, list):
            var_states = np.array(var_states)
            var_states = np.reshape(var_states, (1, -1))

        Mc = Cpm(variables=var_inds,
                 no_child=len(var_inds),
                 C=var_states,
                 p=np.empty(shape=(var_states.shape[0], 1)))

        is_compat = self.iscompatible(Mc, flag=True)
        idx = np.where(is_compat)[0]
        Msubset = self.get_subset(idx, flag)

        return Msubset.p.sum()


    def condition(self, cnd_vars, cnd_states):
        """
        Returns a list of instance of Cpm

        Parameters
        ----------
        cnd_vars: a list of variables to be conditioned
        cnd_states: a list of the states to be conditioned
        """

        assert isinstance(cnd_vars, (list, np.ndarray)), 'invalid cnd_vars'

        if isinstance(cnd_vars, np.ndarray):
            cnd_vars = cnd_vars.tolist()

        if cnd_vars and isinstance(cnd_vars[0], str):
            cnd_vars = self.get_variables(cnd_vars)

        assert isinstance(cnd_states, (list, np.ndarray)), 'invalid cnd_vars'

        if isinstance(cnd_states, np.ndarray):
            cnd_states = cnd_states.tolist()

        if cnd_states and isinstance(cnd_states[0], str):
            cnd_states = [x.values.index(y) for x, y in zip(cnd_vars, cnd_states)]

        Mx = copy.deepcopy(self)

        is_cmp = iscompatible(Mx.C, Mx.variables, cnd_vars, cnd_states)

        C = Mx.C[is_cmp, :].copy()

        _, idx_cnd = ismember(cnd_vars, Mx.variables)
        _, idx_vars = ismember(Mx.variables, cnd_vars)

        Ccond = np.zeros_like(C)
        not_idx_vars = flip(idx_vars)

        if C.size:
            Ccond[:, not_idx_vars] = get_value_given_condn(C, not_idx_vars)

        cnd_vars_m = get_value_given_condn(cnd_vars, idx_cnd)
        cnd_states_m = get_value_given_condn(cnd_states, idx_cnd)
        idx_cnd = get_value_given_condn(idx_cnd, idx_cnd)

        for cnd_var, state, idx in zip(cnd_vars_m, cnd_states_m, idx_cnd):
            try:
                B = cnd_var.B.copy()
            except NameError:
                raise(f'{cnd_var} is not defined')
            else:
                C1 = C[:, idx].copy().astype(int)
                check = [B[x].intersection(B[state]) for x in C1]
                Ccond[:, idx] = [x for x in ismember(check, B)[1]]

        Mx.C = Ccond.copy()

        if Mx.p.size:
            Mx.p = Mx.p[is_cmp]

        if Mx.Cs.size: # NB: ps is not properly updated if the corresponding instance is not in C.

            Cnew_with_minus1 = _get_Cnew(Mx.Cs, np.array([cnd_states]), Mx.variables, cnd_vars, Mx.variables)
            
            mask = np.sum( Cnew_with_minus1 < 0, axis=1 ) < 1
            Cs_new, ps_new = Cnew_with_minus1[mask], Mx.ps[mask]
            q_new, sample_idx_new = Mx.q[mask], Mx.sample_idx[mask]
            
            Mx.Cs, Mx.q, Mx.ps, Mx.sample_idx = Cs_new, q_new, ps_new, sample_idx_new

        return Mx


    def sort(self):

        if self.sample_idx.size:
            idx = argsort(self.sample_idx)
        else:
            idx = argsort(list(map(tuple, self.C[:, ::-1])))

        self.C = self.C[idx, :]

        if self.p.size:
            self.p = self.p[idx]

        if self.q.size:
            self.q = self.q[idx]

        if self.sample_idx.size:
            self.sample_idx = self.sample_idx[idx]


    def get_prob_bnd(self, var_inds, var_states, flag=True, cvar_inds=None, cvar_states=None, cflag=True):

        prob1 = self.get_prob(var_inds, var_states, flag)
        prob_unk = 1.0 - np.sum(self.p) # Unknown probs
        prob1_bnd = [prob1, prob1 + prob_unk]

        if cvar_inds:
           prob2 =  self.get_prob(cvar_inds, cvar_states, cflag)
           prob2_bnd = [prob2, prob2 + prob_unk]
           prob_bnd = [prob1_bnd[0] / prob2_bnd[1], prob1_bnd[1] / prob2_bnd[0]]

        else:
           prob_bnd = prob1_bnd

        prob_bnd[1] = min(1, prob_bnd[1])

        return prob_bnd


    def get_prob_and_cov(self, var_inds, var_states, method='MLE', flag=True, nsample_repeat=0, conf_p=0.95):

        assert isinstance(nsample_repeat, int), 'nsample_repeat must be a nonnegative integer, representing if samples are repeated (to calculate c.o.v.)'

        prob_C = self.get_prob(var_inds, var_states, flag)

        if not nsample_repeat:
            n_round = 1
            nsamp = len(self.Cs)
        else:
            assert len(self.Cs) % nsample_repeat == 0, 'Given number of samples is not divided by given nsample_repeat'
            n_round = int(len(self.Cs) / nsample_repeat)
            nsamp = nsample_repeat

        prob_Cs = 0
        var = 0
        for i in range(n_round):

            row_range = range(i*nsamp, (i + 1)*nsamp)
            is_cmp = iscompatible(self.Cs[row_range,:], self.variables, var_inds, var_states)

            try:
                w = self.ps[row_range] / self.q[row_range]
            except IndexError:
                w = np.ones_like(self.q) # if ps is empty, assume ps is the same as q

            w_ori = w.copy() # weight before compatibility check
            if flag:
                w[~is_cmp] = 0
            else:
                w[is_cmp] = 0

            if method=='MLE':
                mean = w.sum() / nsamp
                prob_Cs += mean

                if np.allclose(w_ori, w_ori[0], atol=1e-4): # this is MCS
                    var1 = np.square(w_ori[0]) * (1 - mean) * mean / nsamp
                    var += var1[0]
                else:
                    var1 = np.square((w - mean) / nsamp)
                    var += var1.sum()

            elif method=='Bayesian':
                neff = len(w_ori)*w_ori.mean()**2 / (sum(x**2 for x in w_ori)/len(w_ori)) # effective sample size
                w_eff = w / w_ori.sum() *neff
                nTrue = w_eff.sum()

                # to avoid numerical errors
                if np.isnan(nTrue):
                    nTrue = 0.0
                if np.isnan(neff[0]):
                    neff[0] = 0.0

                try:
                    a, b = a + nTrue, b + (neff[0] - nTrue)
                except NameError:
                    prior = 0.01
                    a, b = prior + nTrue, prior + (neff[0] - nTrue)

        if method == 'MLE':
            prob = prob_C + (1 - self.p.sum()) * prob_Cs
            cov = (1 - self.p.sum()) * np.sqrt(var) / prob

            # confidence interval
            z = norm.pdf(1 - (1 - conf_p)*0.5) # for both tails
            prob_Cs_int = prob_Cs + z * np.sqrt(var) * np.array([-1, 1])
            cint = prob_C + (1 - self.p.sum()) * prob_Cs_int

        elif method == 'Bayesian':

            mean = a / (a + b)
            var = a*b / (a+b)**2 / (a+b+1)

            prob = prob_C + (1 - self.p.sum()) * mean
            cov = (1 - self.p.sum()) * np.sqrt(var) / prob

            low = beta.ppf(0.5*(1-conf_p), a, b)
            up = beta.ppf(1 - 0.5*(1-conf_p), a, b)
            cint = prob_C + (1 - self.p.sum()) * np.array([low, up])

        return prob, cov, cint


    def get_prob_and_cov_cond(self, var_inds, var_states, cvar_inds, cvar_states, nsample_repeat=0, conf_p=0.95):
        # Assuming beta distribution (i.e. Bayeisan inference)

        assert isinstance(nsample_repeat, int), 'nsample_repeat must be a nonnegative integer, representing if samples are repeated (to calculate c.o.v.)'

        if not nsample_repeat:
            n_round = 1
            nsamp = len(self.Cs)
        else:
            assert len(self.Cs) % nsample_repeat == 0, 'Given number of samples is not divided by given nsample_repeat'
            n_round = int(len(self.Cs) / nsample_repeat)
            nsamp = nsample_repeat

        for i in range(n_round):
            row_range = range(i*nsamp, (i + 1)*nsamp)

            try:
                w_ori = self.ps[row_range] / self.q[row_range] # weight before compatibility check
            except IndexError:
                w_ori = np.ones_like(self.q) # if ps is empty, assume ps is the same as q

            is_cmp1 = iscompatible(self.Cs[row_range,:], self.variables, var_inds, var_states)
            w1 = w_ori.copy()
            w1[~is_cmp1] = 0

            is_cmp2 = iscompatible(self.Cs[row_range,:], self.variables, cvar_inds, cvar_states)
            w2 = w_ori.copy()
            w2[~is_cmp2] = 0

            neff = len(w_ori)*w_ori.mean()**2 / (sum(x**2 for x in w_ori)/len(w_ori)) # effective sample size

            w1_eff = w1 / w_ori.sum() *neff
            nTrue1 = w1_eff.sum()

            w2_eff = w2 / w_ori.sum() *neff
            nTrue2 = w2_eff.sum()

            try:
                a1, b1 = a1 + nTrue1, b1 + (neff[0] - nTrue1)
                a2, b2 = a2 + (nTrue2-nTrue1), b2 + (neff[0] - (nTrue2-nTrue1))
            except NameError:
                prior = 0.01
                a1, b1 = prior + nTrue1, prior + (neff[0] - nTrue1)
                a2, b2 = prior + (nTrue2-nTrue1), prior + (neff[0] - (nTrue2-nTrue1))

        prob_C = self.get_prob(var_inds, var_states)
        prob_C_c = self.get_prob(cvar_inds, cvar_states)

        prob, std, cint = utils.get_rat_dist( prob_C, prob_C_c - prob_C, 1 - self.p.sum(), a1, a2, b1, b2, conf_p )

        cov = std/prob

        return prob, cov, cint
    
def C_prod_C(M1, M2, new_vars):
    """
    Product of two C matrices (i.e. not Cs matrices)
    """
    Cnew_with_minus1 = _get_Cnew(M1.C, M2.C, M1.variables, M2.variables, new_vars)

    n_row1, n_row2 = len(M1.C), len(M2.C)
    pnew = np.repeat(M1.p, n_row2) * np.tile(M2.p, (n_row1, 1)).flatten()

    mask = np.sum( Cnew_with_minus1 < 0, axis=1 ) < 1
    Cnew, pnew = Cnew_with_minus1[mask], pnew[mask]

    return Cnew, pnew

def C_prod_Cs(M1, M2, new_vars):
    """
    Product of C (M1) and Cs (M2) matrices
    """
    Cnew_with_minus1 = _get_Cnew(M1.C, M2.Cs, M1.variables, M2.variables, new_vars)

    n_row1, n_row2 = len(M1.C), len(M2.Cs)
    ps_new = np.repeat(M1.p, n_row2) * np.tile(M2.ps, (n_row1, 1)).flatten()
    sample_idx_new = np.tile(M2.sample_idx, (n_row1, 1))
    q_new = np.repeat(M1.p, n_row2) * np.tile(M2.q, (n_row1, 1)).flatten()

    mask = np.sum( Cnew_with_minus1 < 0, axis=1 ) < 1
    Cs_new, ps_new = Cnew_with_minus1[mask], ps_new[mask]
    q_new, sample_idx_new = q_new[mask], sample_idx_new[mask]

    # Sort rows by sample_idx for readability
    sorted_idx = np.argsort(sample_idx_new.flatten())
    Cs_new, q_new = Cs_new[sorted_idx], q_new[sorted_idx]
    ps_new, sample_idx_new = ps_new[sorted_idx], sample_idx_new[sorted_idx]

    return Cs_new, q_new, ps_new, sample_idx_new

def Cs_prod_Cs(Cs1, Cs2, vars1, vars2, q1, q2, ps1, ps2, sample_idx1, sample_idx2, new_vars):
    """
    Product of two Cs matrices
    """

    # Give warnings if there is a mismatch in sample indices between Cs1 and Cs2
    # Mismatching indices will be ignored
    unique_sidx_M1 = np.unique(sample_idx1)
    unique_sidx_M2 = np.unique(sample_idx1)

    only_in_M1 = unique_sidx_M1 - unique_sidx_M2
    only_in_M2 = unique_sidx_M2 - unique_sidx_M1

    if only_in_M1.size or only_in_M2.size:
        warnings.warn(
            f"Mismatch in unique sample indices between M1 and M2.\n"
            f"Indices only in M1: {sorted(only_in_M1)}\n"
            f"Indices only in M2: {sorted(only_in_M2)}",
            UserWarning
        )

    new_vars_tf1, new_vars_idx1 = ismember( new_vars, vars1 )
    new_vars_tf2, new_vars_idx2 = ismember( new_vars, vars2 )    

    # Common variables
    com_vars_idx1, com_vars_idx2 = [], []
    for tf1, tf2, idx1, idx2 in zip(new_vars_tf1, new_vars_tf2, new_vars_idx1, new_vars_idx2):
        if tf1 and tf2:
            com_vars_idx1.append(idx1)
            com_vars_idx2.append(idx2)

    Cs_new = np.array([], dtype=int).reshape(0, len(new_vars))
    qs_new = np.array([])
    ps_new = np.array([])
    sample_idx_new = np.array([])

    # Mapping Cs_new's columns to Cs1's and Cs2's columns
    # Cs_col_map[0] is the index of Cs1's columns
    # Cs_col_map[1] is the index of Cs2's columns
    Cs_col_map = [[], []]
    for i, (tf1, tf2, idx1, idx2) in enumerate(zip(new_vars_tf1, new_vars_tf2, new_vars_idx1, new_vars_idx2)):
        if tf1:
            Cs_col_map[0].append([i, idx1])
        elif tf2:
            Cs_col_map[1].append([i, idx2])

    for c1, q1, ps1, sidx1 in zip(Cs1, q1, ps1, sample_idx1):
        row_idx2 = np.where(sample_idx2 == sidx1)[0]

        for i2 in row_idx2:
            if Cs2[i2][com_vars_idx2] == c1[com_vars_idx1]:
                cs_new = np.zeros((1, len(new_vars)))
                for i, idx in Cs_col_map[0]:
                    cs_new[0, i] = c1[idx]
                for i, idx in Cs_col_map[1]:
                    cs_new[0, i] = Cs2[i2][idx]

                Cs_new = np.vstack([Cs_new, cs_new])    
                qs_new = np.append(qs_new, q1*q2[i2])
                ps_new = np.append(ps_new, ps1*ps2[i2])
                sample_idx_new = np.append(sample_idx_new, sidx1)

    Cs_new.astype(int)
    sample_idx_new.astype(int)

    return Cs_new, qs_new, ps_new, sample_idx_new


def _get_Cnew(C1, C2, vars1, vars2, new_vars):
    """
    For internal use in "product" method
    """

    new_vars_tf1, new_vars_idx1 = ismember( new_vars, vars1 )
    new_vars_tf2, new_vars_idx2 = ismember( new_vars, vars2 )

    n_row1 = len(C1)
    n_row2 = len(C2)

    Bst_dict = {}
    for k, v in enumerate(new_vars):

        if new_vars_tf1[k] and new_vars_tf2[k]: # common variable to product

            n_val = len(v.values)

            Bvec1 = np.zeros((n_row1, n_val), dtype=int)
            Bvec2 = np.zeros((n_row2, n_val), dtype=int)

            C1_ = C1[:, new_vars_idx1[k]]
            for i, b in enumerate(C1_):
                st = list( v.get_set(b) )
                Bvec1[i, st] = 1
            Bvec1 = np.tile(Bvec1, (n_row2, 1, 1))

            C2_ = C2[:, new_vars_idx2[k]]
            for i, b in enumerate(C2_):
                st = list( v.get_set(b) )
                Bvec2[i, st] = 1
            Bvec2 = np.tile(Bvec2[:, np.newaxis, :], (1, n_row1, 1))

            Bvec = Bvec1 * Bvec2
            Bst = v.get_Bst_from_Bvec( Bvec )
            
            Bst_dict[v.name] = Bst

        elif new_vars_tf1[k]: # it is in self only
            C1_ = C1[:, new_vars_idx1[k]]
            Bst = np.tile(C1_.T, (n_row2, 1, 1))
            Bst = np.squeeze(Bst)

            Bst_dict[v.name] = Bst

        else: # it is in M only
            C2_ = C2[:, new_vars_idx2[k]]
            Bst = np.repeat(C2_.reshape(-1,1), n_row1, axis=1)

            Bst_dict[v.name] = Bst

    Cnew_list = []
    for v in new_vars:
        Cnew_list.append( Bst_dict[v.name].T.reshape(-1) )

    # Cnew with -1
    # -1 elements mean incompatible rows between the two CPMs
    Cnew_with_minus1 = np.column_stack( Cnew_list ) 

    return Cnew_with_minus1


def argsort(seq):

    #http://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python/3382369#3382369
    return sorted(range(len(seq)), key=seq.__getitem__)


def ismember(A, B):
    """
    A: vector
    B: list
    return
     Lia: logical true and false where data in A is found in B
     Lib: the list (same length as A) of index of the first matching elment in B or False for non-matching element

    """

    if isinstance(A, np.ndarray) and (A.ndim > 1):

        assert A.shape[1] == np.array(B).shape[1]

        res = []
        for x in A:
            v = np.where((np.array(B) == x).all(axis=1))[0]
            if len(v):
                res.append(v[0])
            else:
                res.append(False)

    elif isinstance(A, list) and isinstance(B, list):

        res  = [B.index(x) if x in B else False for x in A]

    else:

        if isinstance(B, np.ndarray) and (B.ndim > 1):
            assert len(A) == B.shape[1]

        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            res  = [np.where(np.array(B)==x)[0].min()
                 if x in B else False for x in A]

    lia = [False if x is False else True for x in res]

    return lia, res


def setdiff(first, second):
    """
    matlab setdiff equivalent
    """
    second = set(second)
    first = list(dict.fromkeys(first))
    val = [item for item in first if item not in second]
    iv = [first.index(x) for x in val]
    return val, iv


def get_value_given_condn(A, condn):

    if isinstance(A, np.ndarray) and A.ndim==2 and A.shape[1] == len(condn):
        A = A.T
        val = np.array([x for (i, x) in zip(condn, A) if i is not False])
        if val.size:
            val = val.reshape(-1, A.shape[1]).T
    else:
        assert len(A) == len(condn), f'len of {A} is not equal to len of {condn}'
        val = [x for (i, x) in zip(condn, A) if i is not False]

    return val


def iscompatible(C, variables, check_vars, check_states, composite_state=False):
    """
    Returns a boolean list

    Parameters
    ----------
    C: np.ndarray
    variables: array_like
    check_vars: array_like list of Variable or string
    check_sates: array_like list of index or string
    composite_state: False if the same rows are returned;
        True if composite states and basic state can be
        considered compatible if they are. 
    """
    if check_vars and isinstance(check_vars[0], str):
        check_vars = [x for y in check_vars for x in variables if x.name == y]

    _, idx = ismember(check_vars, variables)
    check_vars = get_value_given_condn(check_vars, idx)

    if len(check_vars) > 0:
        check_states = get_value_given_condn(check_states, idx)
        idx = get_value_given_condn(idx, idx)

        C = C[:, idx].copy()

        for i, (variable, state) in enumerate(zip(check_vars, check_states)):

            if isinstance(state, str):
                state = variable.values.index(state)
                check_states[i] = state

        if not composite_state:
            is_cmp = np.all(C == check_states, axis=1).tolist()

        else:
            Cnew_with_minus1 = _get_Cnew(C, np.array([check_states]), check_vars, check_vars, check_vars)
            is_cmp = np.all(Cnew_with_minus1 >= 0, axis=1).tolist()
    else:
        is_cmp = np.ones(shape=C.shape[0], dtype=bool).tolist()

    return is_cmp


def flip(idx):
    """
    boolean flipped
    Any int including 0 will be flipped False
    """
    return [True if x is False else False for x in idx]


def product(cpms):
    """
    return an instance of Cpm

    cpms: a list or dict of instances of Cpm
    """
    assert isinstance(cpms, (list,  dict)), 'cpms should be a list or dict'

    if isinstance(cpms, dict):
        cpms = list(cpms.values())

    assert cpms, f'{cpms} is empty list'

    prod = cpms[0]
    for cx in cpms[1:]:
        prod = prod.product(cx)

    return prod


def get_prod(A, B):
    """
    A: matrix
    B: matrix
    """
    if len(A.shape) < 2:
        A = np.reshape(A, (A.shape[0], 1))

    if len(B.shape) < 1:
        B=np.reshape(B,(1,))

    assert A.shape[1] == B.shape[0]

    prod_sign = np.sign(A * B)
    prod_val = np.exp(np.log(np.abs(A)) + np.log(np.abs(B)))

    return prod_sign * prod_val

