""" Clock models implementation """

# import numba            as nb
import numpy            as np
import quspin.operators as qo

from .core import ClockCore, _no_check
from .op   import clock2_op, make_clock_op

class Clock2(ClockCore):
    """Two-state clock model (aka Ising model)"""
    def __init__(self, size, pbc):
        """
        Return a two-state clock model (Ising)

        Parameters
        ----------
        size : int
            size of the lattice (number of sites)
        pbc : boolean
            periodic boundary condition
        """
        super().__init__(size=size, pbc=pbc, sps=2, op=clock2_op, allowed_ops="xz")

    def hamiltonian(self, kin=1.0, transv=0.0, long=0.0, dtype=np.float64):
        """
        Return an Hamiltonian of the form
            H = \sum_i (  g_kin Z_i Z_{i+1} + g_transv X_i + g_long Z_i )

        Parameters
        ----------
        kin : float
            kinetic/hopping parameter (g_kin)
        transv : float
            transversal field coupling (g_transv)
        long : float
            longitudinal field coupling (g_long)
        dtype : dtype (default np.float64)
            datatype for the Hamiltonian elements

        Complex couplings are not supported for the `Clock2` class

        Return
        ----------
        H : quspin.operators.hamiltonian 
        """
        if self.pbc:
            kinetic_list = [[kin, i, (i+1) % self.size] for i in range(self.size)]
        else:
            kinetic_list = [[kin, i, i+1] for i in range(self.size-1)]
        transv_list = [[transv, i] for i in range(self.size)]
        long_list   = [[long, i]   for i in range(self.size)]
        static_list = [["zz", kinetic_list], ["x", transv_list], ["z", long_list]]

        return qo.hamiltonian(static_list, [], basis=self.qbasis, dtype=dtype, **_no_check)


def make_Clock(N):
    """
    Return the N-state clock model class (for N=2 use directly the `Clock2` class)

    Parameters
    ----------
    N : int
        dof per site of the clock model

    Return
    ----------
    Clock : class
        a class (like `Clock2`) that represents an N-state quantum clock model.
        The returned class has the following operator set:
          x: forward permutation    |s> -> |s+1 mod N>
          X: backward permutation   |s> -> |s-1 mod N>
          z: position operator      |s> -> |s> * phase**s
          Z: adj position operator  |s> -> |s> * phase**(-s)
        the operators with the uppercase (lowercase) letter are the adjoint 
        of the corresponding operator with lowercase (uppercase) letter.
    """
    clock_op = make_clock_op(N)
    class Clock(ClockCore):
        def __init__(self, size, pbc):
            """
            Return a N-state clock model (Ising)

            Parameters
            ----------
            size : int
                size of the lattice (number of sites)
            pbc : boolean
                periodic boundary condition
            """
            super().__init__(size=size, pbc=pbc, sps=N, op=clock_op, allowed_ops="XxZz")

        def hamiltonian(self, kin=1.0, transv=0.0, long=None, dtype=np.complex128):
            """
            Return an Hamiltonian of the form
                H = \sum_i (  g_kin Z_i Z_{i+1} + g_transv X_i + g_long Z_i + H.c.)

            Parameters
            ----------
            kin : float or complex(float)
                kinetic/hopping parameter (g_kin)
            transv : float or complex(float)
                transversal field coupling (g_transv)
            long : float or complex(float)
                longitudinal field coupling (g_long)
            dtype : dtype (default np.complex128)
                datatype for the Hamiltonian elements

            Return
            ----------
            H : quspin.operators.hamiltonian 
            """
            if self.pbc:
                kinetic_list    = [[kin,          i, (i+1) % self.size] for i in range(self.size)]
                kinetic_hc_list = [[np.conj(kin), i, (i+1) % self.size] for i in range(self.size)]
            else:
                kinetic_list    = [[kin,          i, i+1] for i in range(self.size-1)]
                kinetic_hc_list = [[np.conj(kin), i, i+1] for i in range(self.size-1)]
            transv_list    = [[transv,          i] for i in range(self.size)]
            transv_hc_list = [[np.conj(transv), i] for i in range(self.size)]
            static_list = [["Zz", kinetic_list], ["zZ", kinetic_hc_list],
                           ["x", transv_list],   ["X", transv_hc_list]]
                           # ["z", long_list],     ["Z", long_hc_list]]
            if long is not None:
                long_list   = [[long, i]   for i in range(self.size)]
                long_hc_list   = [[np.conj(long), i]   for i in range(self.size)]
                static_list.append(["z", long_list])
                static_list.append(["Z", long_hc_list])

            return qo.hamiltonian(static_list, [], basis=self.qbasis, dtype=dtype, **_no_check)

    return Clock
