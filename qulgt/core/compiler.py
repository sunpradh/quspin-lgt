import numba as nb
import numpy as np
import quspin.basis.user as qbu

class Compiler(object):

    def __init__(self, dtype, op_fn, allowed_ops):
        self.set_compile_flags(dtype)
        self.op_fn = op_fn
        self.allowed_ops = allowed_ops
        self.numba_compile()

    def set_compile_flags(self, dtype):
        """Set the signature for the CFunc needed for quspin"""
        if dtype == np.uint32:
            self.dtype          = dtype
            self.next_state_sig = qbu.next_state_sig_32
            self.op_sig         = qbu.op_sig_32
        elif dtype == np.uint64:
            self.dtype          = dtype
            self.next_state_sig = qbu.next_state_sig_64
            self.op_sig         = qbu.op_sig_64
        else:
            raise RuntimeError("Cannot determine the signature for numba-compiled functions")

    def numba_compile(self):
        """Numba-compile the required CFunc objects"""
        self.next_state_cfunc = nb.cfunc(self.next_state_sig)(next_state)
        self.op_cfunc = nb.cfunc(self.op_sig)(self.op_fn)
        self.op_dict  = dict(op=self.op_cfunc, op_args=np.array([], dtype=self.dtype))

    def compile(self, N, sps, states):
        """Construct the user basis"""
        self.states    = states
        self.pcon_dict = self.get_pcon_dict(states)
        return qbu.user_basis(
            N           = N,
            basis_dtype = self.dtype,
            op_dict     = self.op_dict,
            allowed_ops = self.allowed_ops,
            sps         = sps,
            pcon_dict   = self.pcon_dict
        )

    # the following two functions has to be here otherwise quspin will not work
    # properly for some reasom. Probably related to the objects life-time
    def get_Ns(self, N, Np):
        return self.states.size

    def get_s0(self, N, Np):
        return self.states[0]

    def get_pcon_dict(self, states):
        return dict(
            Np=(),
            next_state      = self.next_state_cfunc,
            next_state_args = states,
            get_s0_pcon     = self.get_s0,
            get_Ns_pcon     = self.get_Ns
        )

def next_state(s, counter, N, args):
    return args[counter+1]
