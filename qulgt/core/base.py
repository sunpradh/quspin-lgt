"""
This module implements the base class through which the LGT model interfaces
with the QuSpin class quspin.basis.user_basis
"""

from abc import ABC, abstractclassmethod
from .lattice  import Lattice
from .compiler import Compiler
from .drawing  import StateDrawer


class GaugeTheoryBase(Lattice, ABC):
    """
    Base class for all the LGT models.

    It manages all the administrative stuff about the LGTs classes, except for the mkstates function
    """

    def __init__(self, size, pbc, spl, op_fn, allowed_ops, **kwargs):
        # initialize the lattice
        super().__init__(size, pbc, spl)
        self._compiler = Compiler(dtype=self.dtype, op_fn=op_fn, allowed_ops=allowed_ops)
        # Any model can be constructed by costumizing the _mkstates function
        self.states = self._mkstates(**kwargs)
        self.qbasis = self._compiler.compile(N=self.nlinks, sps=self.spl, states=self.states)
        self._state_drawer = StateDrawer(self)

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
        return self._state_drawer.int_repr(state)

    def draw_state(self, state):
        """
        Given the integer representation of the lattice state, it returns a string of
        a diagram representing the state
        """
        return self._state_drawer.draw_state(state)

    def list_states(self, *args, **kwargs):
        """
        List the desired states. Can be used as:

            str_states(stop)
            str_states(start, stop)
            str_states(start, stop, step)
        """
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
        return Lattice.__str__(self)

    def lattice_repr(self):
        return Lattice.__repr__(self)
