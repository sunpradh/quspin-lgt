import numba as nb
import numpy as np

from .znbase import ZnBase

def make_ZN_op(N):
    """Generate the operator set of a ZN theory with given N"""
    def zn_op(op_struct_ptr, op_str, ind, nlinks, args):
        op_struct = nb.carray(op_struct_ptr, 1)[0]
        err = 0

        # site index
        ind = nlinks - ind - 1
        # link state (0,...,N-1)
        # s = zn_occ(op_struct.state, ind, sps=N)
        s = (op_struct.state // N**ind) % N
        # site mask
        b = N**ind
        # complex phase
        omega = np.exp(2j * np.pi / N)

        # Operators: U ('U'), Udag ('u'), V ('V'), Vdag ('v')
        if op_str == 85:  # U
            op_struct.state += b if s < N-1 else -(N-1)*b
        elif op_str == 117:  # u
            op_struct.state -= b if s > 0   else -(N-1)*b
        elif op_str == 86:  # V
            op_struct.matrix_ele *= omega**s
        elif op_str == 118:  # v
            op_struct.matrix_ele *= omega**(-s)
        else:
            op_struct.matrix_ele = 0
            err = -1

        return err

    return zn_op


class ZN(ZnBase):
    def __init__(self, N, size, pbc=(True, True), sector='all'):
        """
        Create an object of the ZN pure gauge model, with the given lattice geometry.
        The topological sector and the dimension of the local Hilbert space can be choosen.

        Parameters
        ----------
        size : tuple(int, int)
            number of sites along the x- and y-axis respectively.
            This define the size of the lattice
        pbc : tuple(bool, bool) (default (True,True))
            periodic boundary conditions along x and y, respectively.
            If True then it will implements periodic boundary along that direction.
        sector : tuple(int, int) or 'all' (default: 'all')
            specify in which topological sector to work.
            The ints can go from 0 to N-1.

        """
        super().__init__(
            size=size,
            pbc=pbc,
            spl=N,
            sector=sector,
            op_fn=make_ZN_op(N),
            allowed_ops="UuVv"
            )
        self.modelname = 'Z' + str(N)
        if N == 3:
            self._state_drawer.update_char_table(
                hlinks=[' ', '→', '←'],
                vlinks=[' ', '↑', '↓']
                )

    def __repr__(self):
        return f'<{self.modelname} on {self.lattice_repr()}, {self.Ns} states, sector {self.sector}>'

