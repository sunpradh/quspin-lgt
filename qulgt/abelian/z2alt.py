"""
This module implements the Z2 Lattice Gauge Theory, alternative version.
Made for Luca
"""

import numba as nb
from .znbase import ZnBase

def z2alt_op(op_struct_ptr, op_str, ind, nlinks, args):
    """
    Operator set for the Z2 case. To be used with quspin.basis.user.user_basis
    """
    op_struct = nb.carray(op_struct_ptr, 1)[0]
    err = 0
    # Quspin convention for bits to sites
    # the left-most bit represent the first site
    ind = nlinks - ind - 1
    # WARNING: in _link_mx the links are numbered from 1 to nlinks
    # Quspin only accept integers from 0 to nlinks-1 for the sites

    # link state (0 or 1)
    s = (op_struct.state >> ind) & 1
    # site mask
    b = (1 << ind)

    # the only operator we need for now are U and V
    # U = |+><-| + |-><+|
    # V = |-><-| - |+><+|
    if op_str == 85:  # U
        # it flips the state
        op_struct.state ^= b
    elif op_str == 86:  # V
        # 0 for |->, 4 for |+>
        op_struct.matrix_ele *= 4*s
    else:
        # you fucked up
        op_struct.matrix_ele = 0
        err = -1

    return err


class Z2lumia(ZnBase):
    def __init__(self, size, pbc=(True, True), sector='all'):
        """
        Create an object of the Z2 pure gauge model, with the given lattice geometry.
        The topological sector can be choosen.

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
            The ints can be only 0 or 1.
        """
        super().__init__(
                size=size,
                pbc=pbc,
                spl=2,
                sector=sector,
                op_fn=z2alt_op,
                allowed_ops="uUvV"
            )
        self._state_drawer.update_char_table(hlinks=[' ', '-'], vlinks=[' ', '|'])
        self.modelname = 'Z2'

    def __repr__(self):
        return f'<{self.modelname} on {self.lattice_repr()}, {self.Ns} states, sector {self.sector}>'
