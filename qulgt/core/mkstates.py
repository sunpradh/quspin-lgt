import numpy as np
import numba as nb

# numba signatures
_occ_sig = [nb.uint32(nb.uint32, nb.uint32), nb.uint64(nb.uint64, nb.uint64)]
_mkstates_sig = [
        nb.uint32[:](nb.uint32, nb.uint32, nb.uint32[:,:]),
        nb.uint64[:](nb.uint64, nb.uint64, nb.uint64[:,:])
                ]

# Z2 helpers
#
# The Z2 part is separated from the others Zn because it can be
# optimized with bitwise operations

@nb.njit(_occ_sig, inline="always")
def z2_occ(s, n):
    """Z2 occupation number (0 or 1)"""
    return (s >> n) & 1

# Zn Helpers

@nb.njit
def zn_occ(state, n, spl):
    """Zn occupation number (0 ... sps-1)"""
    return (state // spl**n) % spl

@nb.njit
def zn_U_op(state, n, spl):
    """Zn comparator in the electric base"""
    state += (spl**n) if zn_occ(state, n, spl) < (spl-1) else -(spl-1)*(spl**n)
    return state

@nb.njit
def zn_Udag_op(state, n, spl):
    """Zn adjoint comparator in the electric base"""
    state -= (spl**n) if zn_occ(state, n, spl) > 0 else -(spl-1)*(spl**n)
    return state

@nb.njit
def zn_plaquette_op(state, plaquette, sps):
    """Zn plaquette operator in the electric base"""
    state = zn_U_op(state, plaquette[0], sps)
    state = zn_U_op(state, plaquette[1], sps)
    state = zn_Udag_op(state, plaquette[2], sps)
    state = zn_Udag_op(state, plaquette[3], sps)
    return state


#
# Generating function for the Hilbert space
#
def get_mkstates_fn(spl):
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
        for flux in nb.prange(0, nstates):
            for n in range(plaquettes.shape[0]):
                if (flux >> n) & 0x1:
                    # the reason for this loop is because numba has some problems with slicing
                    for l in range(4):
                        states[flux] ^= (0x1 << plaquettes[n,l])
        return states

    # Zn case
    @nb.njit(_mkstates_sig)
    def mkstates_zn(vacuum, nstates, plaquettes):
        n_plaq = plaquettes.shape[0]
        states = np.full(nstates, vacuum, dtype=plaquettes.dtype)
        # same reasoning of mkstates_z2 but in base n
        # therefore we cannot use bitwise operations
        for flux in nb.prange(0, nstates):
            for n in range(n_plaq):
                flux_in_n = (flux // spl**n) % spl
                for _ in range(flux_in_n):
                    states[flux] = zn_plaquette_op(states[flux], plaquettes[n,:], spl)
        return states

    return mkstates_z2 if spl == 2 else mkstates_zn


# Loop operator
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

