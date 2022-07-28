"""
Quantum clock model base class
"""
# import numba as nb
import numpy as np
import numba as nb
import quspin.basis.user as qbu


class ClockCore(object):
    """
    Base class for the clock model. 

    It just implement basic functionality for interfacing with QuSpin user_basis.
    No need to use it directly
    """
    def __init__(self, size, pbc, sps, op, allowed_ops):
        self.modelname = 'ClockCore'
        self.size = size
        self.pbc = pbc
        self.sps = sps
        self._set_dtype()
        self.states = np.arange(self.sps**self.size, dtype=self.dtype)
        self.qbasis = self._compile(op, allowed_ops)

    def _set_dtype(self):
        """Infer the dtype to use throug out the class"""
        if self.sps**self.size <= 2**32:
            self.dtype = np.uint32
            self._next_state_sig = qbu.next_state_sig_32
            self._op_sig = qbu.op_sig_32
        elif self.sps**self.size <= 2**64:
            self.dtype = np.uint64
            self._next_state_sig = qbu.next_state_sig_64
            self._op_sig = qbu.op_sig_64
        else:
            raise RuntimeError(f"Lattice with size={self.size} with the spl={self.spl} is too big for uint64")

    # the following two functions has to be here otherwise quspin will not work
    # properly for some reasom. Probably related to the objects life-time
    def _get_Ns(self, N, Np):
        return self.states.size

    def _get_s0(self, N, Np):
        return self.states[0]

    def _get_pcon_dict(self):
        return dict(
            Np=(),
            next_state      = self._next_state,
            next_state_args = self.states,
            get_s0_pcon     = self._get_s0,
            get_Ns_pcon     = self._get_Ns
        )

    def _compile(self, op, allowed_ops):
        """Numba-compile the required CFunc objectes and feed them correctly to quspin"""
        self._op         = nb.cfunc(self._op_sig)(op)
        self._next_state = nb.cfunc(self._next_state_sig)(_next_state)
        self._op_dict    = dict(op=self._op, op_args=np.array([], dtype=self.dtype))
        self._pcon_dict  = self._get_pcon_dict()
        
        return qbu.user_basis(
            N           = self.size,
            basis_dtype = self.dtype,
            op_dict     = self._op_dict,
            allowed_ops = allowed_ops,
            sps         = self.sps,
            pcon_dict   = self._pcon_dict
        )


def _next_state(s, counter, N, args):
    return args[counter+1]

_no_check = dict(check_symm=False, check_pcon=False, check_herm=False)
