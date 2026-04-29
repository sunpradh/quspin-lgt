"""
This module implements the base class through which the LGT model interfaces
with the QuSpin class quspin.basis.user_basis
"""

import numpy as np
from abc    import ABC, abstractclassmethod
from typing import TypeVar

from .base_lattice   import LatticeT
from .square_lattice import SquareLattice
from .compiler       import Compiler
from .drawing        import StateDrawer

from quspin.basis.user import op_sig_32, op_sig_64

op_fn_type = op_sig_32 | op_sig_64

class GaugeTheoryBase(ABC):
    """
    Base class for all the LGT models.

    It manages all the administrative stuff about the LGTs classes and QuSpin, except for the `mkstates` function
    """

    def __init__(self,
                 lattice: LatticeT,
                 spl: int,
                 op_fn: op_fn_type,
                 allowed_ops: str,
                 **kwargs):
        """
        Parameters
        ----------
        lattice: LatticeT
            An instance of a lattice-type classe, can be either `SquareLattice` or `TriangularLattice`
        spl: int
            Number of states per link
        op_fn: function in the form `op(op_struct_ptr, op_str, site_ind, N, args)`
            See QuSpin documentation at
            `https://quspin.github.io/QuSpin/tutorials/user_basis.html#op-op-struct-ptr-op-str-site-ind-n-args`
        allowed_ops: str
            A string where each character stands for a different operator, i.e. `+`, `-`, `z` etc...
            See again QuSpin documentation for more details
        """
        # initialize the lattice
        self.lattice = lattice
        self.spl = spl
        self.dtype = self._infer_dtype(nlinks=lattice.nlinks, spl=spl)
        self._compiler = Compiler(dtype=self.dtype, op_fn=op_fn, allowed_ops=allowed_ops)
        # Any model can be constructed by costumizing the _mkstates function
        self.states = self._mkstates(**kwargs)
        self.qbasis = self._compiler.compile(N=self.lattice.nlinks, sps=self.spl, states=self.states)
        # WARNING `StateDrawer` only supports square lattices
        if type(lattice) == SquareLattice:
            self._state_drawer = StateDrawer(self, lattice=self.lattice, spl=self.spl)
        else:
            print("WARNING: the drawing of the states is available only for square lattices")
            self._state_drawer = None

    def _infer_dtype(self, nlinks: int, spl: int):
        """Infer the dtype to use through out the class"""
        if spl**nlinks <= 2**32:
            return np.uint32
        elif spl**nlinks <= 2**64:
            return np.uint64
        else:
            raise RuntimeError(f"Lattice with size={self.size} and spl={self.spl} is too big")


    @abstractclassmethod
    def _mkstates(self, **kwargs):
        pass

    def __len__(self):
        """Return the number of states"""
        return self.qbasis.Ns

    @property
    def Ns(self):
        return self.qbasis.Ns

    def __repr__(self):
        return f'<GaugeTheoryBase class on {self.lattice_repr()}>'

    def __str__(self):
        if not self._state_drawer:
            print("WARNING string representation currently not available")
            return ""
        ret = repr(self) + '\n\n------------------------------\n'
        if self.qbasis.Ns > 20:
            ret += self.list_states(10)
            ret += '    . . . . .\n\n'
            ret += self.list_states(self.Ns - 10, self.Ns, header=False)
        else:
            ret += self.list_states(0, self.qbasis.Ns)
        return ret

    def int_repr(self, state):
        """
        Returns a string representation of the given state (in integer repr)
        """
        if not self._state_drawer:
            print("WARNING string representation currently not available")
            return ""
        return self._state_drawer.int_repr(state)

    def draw_state(self, state):
        """
        Given the integer representation of the lattice state, it returns a string of
        a diagram representing the state
        """
        if not self._state_drawer:
            print("WARNING string representation currently not available")
            return ""
        return self._state_drawer.draw_state(state)

    def list_states(self, *args, **kwargs):
        """
        List the desired states. Can be used as:

            str_states(stop)
            str_states(start, stop)
            str_states(start, stop, step)
        """
        if not self._state_drawer:
            print("WARNING string representation currently not available")
            return ""
        header = '   index)   |state>\t int\n'
        if 'header' in kwargs:
            ret = header if kwargs['header'] else '\n'
        else:
            ret = header

        start = 0
        stop  = args[0] if len(args) > 1 else self.qbasis.Ns
        step  = 1
        if len(args) > 1:
            start = args[0]
            stop  = args[1]
            step  = args[2] if len(args) > 2 else 1

        for i, state in enumerate(self.qbasis.states[slice(start, stop, step)]):
            ret += f'{start+i:>8d})   |{self.int_repr(state)}>\t {state:<8d}\n'
            ret += '\n' + self.draw_state(state) + '\n\n'
        return ret

    def lattice_str(self):
        """Get a representation of the lattice in ascii art"""
        # quick and dirty way
        return str(self.lattice)

    def lattice_picture(self):
        """Paint a picture of the lattice"""
        self.lattice.draw()

    def lattice_repr(self):
        return repr(self.lattice)


GaugeTheoryT = TypeVar('GaugeTheoryT', bound=GaugeTheoryBase)
