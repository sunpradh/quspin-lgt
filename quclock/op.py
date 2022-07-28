"""Implementation of the operator sets for the various clock model"""
import numpy as np
import numba as nb

def clock2_op(op_struct_ptr, op_str, ind, size, args):
    """
    Operator set for the 2-Clock model (Ising model). To be used with quspin.basis.user.user_basis
    """
    op_struct = nb.carray(op_struct_ptr, 1)[0]
    err = 0
    # Quspin convention for bits to sites
    # the left-most bit represent the first site
    ind = size - ind - 1

    # link state (-1 or 1)
    # state = (((op_struct.state >> ind) & 1) << 1) - 1
    state = 2 * ((op_struct.state >> ind) & 1) - 1
    # site mask
    mask = (1 << ind)

    # the only operator we need for now are U and V
    # X = |+><-| + |-><+|
    # Z = |-><-| - |+><+|
    if op_str == 120:  # 'x'
        # it flips the state
        op_struct.state ^= mask
    elif op_str == 122:  # 'z'
        # +1 for |->, -1 for |+>
        op_struct.matrix_ele *= state
    else:
        # you fucked up
        op_struct.matrix_ele = 0
        err = -1
    return err


def make_clock_op(N):
    """
    Returns the operator set for the N-state clock model

    Parameters
    ----------
    N : int
        dimension of the local Hilbert space

    Returns
    ----------
    clock_op : function(op_struct_ptr, op_str, ind, size, args)
        to be compiled with numba.CFunc and to be used with quspin.basis.user.user_basis
    """
    def clock_op(op_struct_ptr, op_str, ind, size, args):
        op_struct = nb.carray(op_struct_ptr, 1)[0]
        err = 0
        # Index
        # using QuSpin convention for bits to sites 
        # the left-most bit represent the first site
        ind = size - ind - 1
        # State (s = 0,...,N-1)
        state = (op_struct.state // N**ind) % N
        # site mask
        mask  = N**ind
        # phase of the clock
        phase = np.exp(2j*np.pi / N)

        # Operators
        # x: forward permutation    |s> -> |s+1 mod N>
        # X: backward permutation   |s> -> |s-1 mod N>
        # z: position operator      |s> -> |s> * phase**s
        # Z: adj position operator  |s> -> |s> * phase**(-s)
        if op_str == 120: # x
            op_struct.state += mask if state < N-1 else -(N-1)*mask
        elif op_str == 88: # X
            op_struct.state -= mask if state > 0 else -(N-1)*mask
        elif op_str == 122: # z
            op_struct.matrix_ele *= phase**state
        elif op_str == 90: # z
            op_struct.matrix_ele *= phase**(-state)
        else:
            # you fucked up
            op_struct.matrix_ele = 0
            err = -1
        return err

    return clock_op

