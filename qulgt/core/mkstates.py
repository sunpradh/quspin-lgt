import numpy as np
import numba as nb

from .base_lattice import LatticeT
from .tri_lattice import TriangularLattice
from .square_lattice import SquareLattice

#
# Numba signatures
#

# Occupation number operator
_occ_z2_sig = [nb.uint32(nb.uint32, nb.uint32), nb.uint64(nb.uint64, nb.uint64)]
_occ_zn_sig = [nb.uint32(nb.uint32, nb.uint32, nb.uint32), nb.uint64(nb.uint64, nb.uint64, nb.uint64)]

# U operator (comparator)
_U_op_sig = [
    nb.uint32(nb.uint32, nb.uint32, nb.uint32),
    nb.uint64(nb.uint64, nb.uint64, nb.uint64)
]

# Plaquette operator
_plaq_square_op_sig = [
    nb.uint32(nb.uint32, nb.uint32[:], nb.uint32),
    nb.uint64(nb.uint64, nb.uint64[:], nb.uint64)
]
_plaq_tri_op_sig = [
    nb.uint32(nb.uint32, nb.uint32[:], nb.uint32, nb.uint32),
    nb.uint64(nb.uint64, nb.uint64[:], nb.uint64, nb.uint64)
]

# State generator main function
# Special Zn triangular lattice case
_mkstates_tri_sig = [
    nb.uint32[:](nb.uint32, nb.uint32, nb.uint32[:,:], nb.uint32[:]),
    nb.uint64[:](nb.uint64, nb.uint64, nb.uint64[:,:], nb.uint64[:])
]
# All the others case
_mkstates_sig = [
    nb.uint32[:](nb.uint32, nb.uint32, nb.uint32[:,:]),
    nb.uint64[:](nb.uint64, nb.uint64, nb.uint64[:,:])
]


#
# Z2 Case, accomplished through bitwise operations
#
@nb.njit(_occ_z2_sig, inline="always")
def z2_occ(s, n):
    """Z2 occupation number (0 or 1)"""
    return (s >> n) & 0x1

#
# Zn Case, bitwise operations not applicable
#
@nb.njit(_occ_zn_sig, inline="always")
def zn_occ(state, n, spl):
    """Zn occupation number (0 ... sps-1)"""
    return (state // spl**n) % spl

@nb.njit(_U_op_sig)
def zn_U_op(state, n, spl):
    """Zn comparator in the electric base"""
    state += (spl**n) if zn_occ(state, n, spl) < (spl-1) else -(spl-1)*(spl**n)
    return state

@nb.njit(_U_op_sig)
def zn_Udag_op(state, n, spl):
    """Zn adjoint comparator in the electric base"""
    state -= (spl**n) if zn_occ(state, n, spl) > 0 else -(spl-1)*(spl**n)
    return state

# Square lattice
@nb.njit(_plaq_square_op_sig)
def zn_square_plaq_op(state, plaquette, spl):
    """Zn plaquette operator in the electric base on a square lattice"""
    state = zn_U_op(state, plaquette[0], spl)
    state = zn_U_op(state, plaquette[1], spl)
    state = zn_Udag_op(state, plaquette[2], spl)
    state = zn_Udag_op(state, plaquette[3], spl)
    return state

# Triangular lattice
@nb.njit(_plaq_tri_op_sig)
def zn_tri_plaq_op(state, plaquette, plaq_type, spl):
    """Zn plaquette operator in the electric basis on a triangular lattice"""
    if plaq_type == 0:
        state = zn_U_op(state, plaquette[0], spl)
        state = zn_U_op(state, plaquette[0], spl)
        state = zn_Udag_op(state, plaquette[0], spl)
    elif plaq_type == 1:
        state = zn_U_op(state, plaquette[0], spl)
        state = zn_Udag_op(state, plaquette[0], spl)
        state = zn_Udag_op(state, plaquette[0], spl)
    return state


#
# Generating function for the Hilbert space
#
def mkstates_fn_factory(lattice: LatticeT, spl: int):
    """
    Return the function for generating the states given a local Hilbert dimension

    Parameters
    ----------
    spl : int
        the local Hilbert dimension of the link (states-per-link)

    Returns
    ----------
    mkstates : function(vacuum, nstates, plaquettes)
        The generating function for the Hilbert space given a specified vacuum.
        Parameters of the returned function:
            vacuum : uint
                the starting vacuum
            nstates : int
                number of expected states
            plaquettes : int[:,:]
                nx4 matrix, each row represent the link indices of a plaquette
        Return value:
            states : uint[:]
    """
    # Z2 case
    @nb.njit(_mkstates_sig)
    def mkstates_z2(vacuum, nstates, plaquettes):
        states = np.full(nstates, vacuum, dtype=plaquettes.dtype)
        # every element of states represent a possible combination of the
        # fluxes in integer representation base 2
        n_plaqs = plaquettes.shape[0]
        plaq_size = plaquettes.shape[1]
        for flux in nb.prange(0, nstates):
            for n in range(n_plaqs):
                if z2_occ(flux, n):
                    # the reason for this loop is because numba has some problems with slicing
                    for l in range(plaq_size):
                        states[flux] ^= (0x1 << plaquettes[n,l])
        return states

    # Zn square case
    @nb.njit(_mkstates_sig)
    def mkstates_square_zn(vacuum, nstates, plaquettes):
        n_plaq = plaquettes.shape[0]
        states = np.full(nstates, vacuum, dtype=plaquettes.dtype)
        # same reasoning of `mkstates_z2` but in base n
        # therefore we cannot use bitwise operations
        for flux in nb.prange(0, nstates):
            for n in range(n_plaq):
                flux_in_n = (flux // spl**n) % spl
                for _ in range(flux_in_n):
                    states[flux] = zn_square_plaq_op(states[flux], plaquettes[n, :], spl)
        return states

    # Zn triangular case
    @nb.njit(_mkstates_tri_sig)
    def mkstates_triangular_zn(vacuum, nstates, plaquettes, plaquette_types):
        n_plaq = plaquettes.shape[0]
        states = np.full(nstates, vacuum, dtype=plaquettes.dtype)
        # same as `mkstates_square_zn` but the plaquette operation
        # depend also on the type of plaquette
        for flux in nb.prange(0, nstates):
            for n in range(n_plaq):
                flux_in_n = (flux // spl**n) % spl
                for _ in range(flux_in_n):
                    states[flux] = zn_tri_plaq_op(states[flux], plaquettes[n, :], plaquette_types[n], spl)
        return states

    if spl == 2:
        return mkstates_z2
    elif type(lattice) == SquareLattice:
        return mkstates_square_zn
    elif type(lattice) == TriangularLattice:
        return mkstates_triangular_zn
    else:
        raise RuntimeError(f"Unrecognized lattice type: {lattice}")



class StateGenerator:
    def __init__(self, lattice: LatticeT, spl: int, dtype):
        self.lattice = lattice
        self.spl = spl
        self.dtype = dtype
        if type(self.lattice) == SquareLattice:
            self.plaquettes = np.array(self.lattice.plaquettes(from_zero=True, flip=True), dtype=self.dtype)
        elif type(self.lattice) == TriangularLattice:
            plqs = self.lattice.plaquettes(include_type=True, from_zero=True, flip=True)
            self.plaquette_types = np.array([type_ for _, type_ in plqs], dtype=self.dtype)
            self.plaquettes = np.array([plaq for plaq, _ in plqs], dtype=self.dtype)
        else:
            raise RuntimeError(f"Unrecognized lattice type for {lattice}")
        self.n_plaqs = len(self.plaquettes)
        self.nstates = self.spl**(self.n_plaqs - 1*(self.lattice.pbc_x and self.lattice.pbc_y))
        self._mkstates_fn = mkstates_fn_factory(lattice=self.lattice, spl=self.spl)

    def __call__(self, vacuum: int):
        if type(self.lattice) == TriangularLattice and self.spl != 2:
            return self._mkstates_fn(
                vacuum,
                nstates=self.nstates,
                plaquettes=self.plaquettes,
                plaquette_types=self.plaquette_types
            )
        else:
            return self._mkstates_fn(
                vacuum,
                nstates=self.nstates,
                plaquettes=self.plaquettes,
            )



# Loop operator
# (no idea why it's here)
def get_loop_op_fn(spl):
    """
    Return a function that creates non-contractible loops.
    Useful for creating all the possible vacuums
    """
    @nb.njit([nb.uint32(nb.uint32, nb.uint32[:]), nb.uint64(nb.uint64, nb.uint64[:])])
    def z2_loop(state, loop):
        state ^= (np.uint8(1) << loop).sum()
        return state

    @nb.njit([nb.uint32(nb.uint32, nb.uint32[:]), nb.uint64(nb.uint64, nb.uint64[:])])
    def zn_loop(state, loop):
        for link in loop:
            state = zn_U_op(state, link, spl)
        return state

    return z2_loop if spl == 2 else zn_loop

