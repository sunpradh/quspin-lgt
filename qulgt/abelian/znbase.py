"""
This module implements the Zn lattice gauge theory base class
"""
import numpy as np
from itertools import product
import quspin.operators as qo

from ..core.base     import GaugeTheoryBase
from ..core.mkstates import get_mkstates_fn
from ..utils.iter import zip_nearest

no_check = dict(check_symm=False, check_pcon=False, check_herm=False)

def make_op_str(next, prev, op_str):
    # dir = 0 -> horizontal direction
    # dir = 1 -> vertical direction
    dir = 0 if next[1] == prev[1] else 1
    length = abs(next[dir] - prev[dir])
    return (op_str[0] if next[dir] > prev[dir] else op_str[1]) * length

class ZnBase(GaugeTheoryBase):

    def __init__(self, *args, **kwargs):
        """
        Base class for all pure gauge Zn models.
        Additional kwargs:
            sector : string (default 'all')
        """
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return f'<ZnBase on {self.lattice_repr()}, {self.Ns} states, sector {self.sector}>'

    def _mkstates(self, **kwargs):
        """
        Make all the states for the basis.

        If a sector is specified with a pair of int then only the states
        in that sector are constructed.
        """
        sector = kwargs.get('sector', 'all')

        # list of plaquettes, with flipped integer repr
        plaquettes = self.plaquettes(from_zero=True, flip=True)
        # number of possible states
        n_plq = len(plaquettes)
        nstates = self.spl**(n_plq - 1*(self.pbc_x and self.pbc_y))
        # convert the list to an array
        plaquettes = np.array(plaquettes, dtype=self.dtype)

        mkstates_kwargs = dict(nstates=nstates, plaquettes=plaquettes)
        mkstates_fn = get_mkstates_fn(self.spl)

        if sector != 'all':
            if sector in self._avail_sectors():
                self.sector = sector
                states = mkstates_fn(self._get_sector_vacuum(sector), **mkstates_kwargs)
            else:
                raise RuntimeError(f"The sector {sector} is not valid")
        else:
            self.sector = 'all'
            states = [
                        mkstates_fn(self._get_sector_vacuum(sect), **mkstates_kwargs)
                        for sect in self._avail_sectors()
                     ]
            states = np.array(states, dtype=self.dtype).ravel()
        return states

    def _get_sector_vacuum(self, sector):
        """Vacuum state for a given sector"""
        if sector not in self._avail_sectors():
            raise RuntimeError(f"Specified sector {sector} is not valid")
        vacuum  = self.dtype(0)
        loops = [
            self.path((0,0), (self.Lx, 0), flip=True),
            self.path((0,0), (0, self.Ly), flip=True)
        ]
        for i in range(2):
            for _ in range(sector[i]):
                for l in loops[i]:
                    vacuum += self.dtype(self.spl ** int(l))
        return vacuum

    def _avail_sectors(self):
        x_sectors = range(self.spl if self.pbc_x else 1)
        y_sectors = range(self.spl if self.pbc_y else 1)
        return product(x_sectors, y_sectors)

    def coupling_plaquettes(self, coupling=0.0):
        """
        Return an array of all plaquettes and their coupling.
        Useful for the Hamiltonian construction with QuSpin.

        Parameters
        ----------
        coupling : float or np.array(float) (default: 0.0)
            coupling of the plaquettes

        Return
        ----------
        plq : list
        """
        plqs = self.plaquettes()
        plq_arr = np.zeros((len(plqs), 5))
        plq_arr[:, 1:] = np.array(plqs)
        plq_arr[:, 0 ] = coupling
        return plq_arr.tolist()

    def coupling_links(self, coupling=0.0):
        """
        Return an array of all the links and their coupling.
        Useful for the Hamiltonian construction with QuSpin.

        Parameters
        ----------
        coupling : float or np.array(float) (default: 0.0)
            coupling of the links

        Return
        ----------
        plq : list
        """
        link_array = np.zeros((self.nlinks, 2))
        link_array[:, 0] = coupling
        link_array[:, 1] = np.arange(self.nlinks)
        return link_array.tolist()

    def hamiltonian(self, plq, elec, dtype=np.complex128):
        """
        Return a Hamiltonian (a QuSpin object) with the specified
        coupling.  Only static interaction, no dynamics.

        Parameters
        ----------
        plq : float or np.array
            coupling on the plaquettes term
        elec : float or np.array
            electric field coupling (on the links)
        dtype : numpy.dtype, optional (default: np.complex128)
            dtype of the Hamiltonian. Default: np.complex128

        Return
        ----------
        H : quspin.hamiltonian
            A QuSpin operator that represent the Hamiltonian con
        """
        coupling_plqs       = self.coupling_plaquettes(plq)
        coupling_plqs_conj  = self.coupling_plaquettes(np.conj(plq))
        coupling_links      = self.coupling_links(elec)
        coupling_links_conj = self.coupling_links(np.conj(elec))
        slist = [
            ["uuUU", coupling_plqs],      # U_pql
            ["UUuu", coupling_plqs_conj], # U_plq^dag
            ["V",    coupling_links],     # V
            ["v",    coupling_links_conj] # Vdag
        ]
        return qo.hamiltonian(
                    slist, # static list
                    [], # empty dynamic list (no time dependency)
                    basis=self.qbasis,
                    dtype=dtype,
                    **no_check # do not perform any checks
                )

    def quantum_operator(self, plq, elec, dtype=np.complex128, which='ham'):
        """
        Return the quspin.operators.quantum_operator for the hamiltonian

        Parameters
        ----------
        plq : float or np.array (default 1)
            coupling on plaquettes
        elec : float or np.array (default 1)
            electric coupling on the links
        which : 'ham', 'quantop' or 'linop'
            how to return the quantum_operator object
                'ham' : as an quspin.operators.hamiltonian
                'quantop' : as an quspin.operators.quantum_operator
                'linop' : as a scipy.sparse.linalg.Linearquantum_operator
        """
        if not hasattr(self, '_quantum_operator'):
            plq_list      = [["uuUU", self.coupling_plaquettes(1.0)         ]]
            plq_list_dag  = [["UUuu", self.coupling_plaquettes(np.conj(1.0))]]
            elec_list     = [["V",    self.coupling_links(1.0)             ]]
            elec_list_dag = [["v",    self.coupling_links(np.conj(1.0))    ]]
            input_dict = dict(
                plq      = plq_list,
                plq_dag  = plq_list_dag,
                elec     = elec_list,
                elec_dag = elec_list_dag,
            )
            self._quantum_operator = qo.quantum_operator(input_dict, N=self.nlinks, basis=self.qbasis, dtype=dtype, **no_check)

        if which == 'quantop':
            return self._quantum_operator
        else:
            couplings=dict(plq=plq, plq_dag=np.conj(plq), elec=elec, elec_dag=np.conj(elec))
            if which == 'ham':
                return self._quantum_operator.tohamiltonian(couplings)
            elif which == 'linop':
                return self._quantum_operator.aslinearoperator(couplings)
            else:
                raise RuntimeError(f'Unrecognized option which="{which}"')

    def wilson_loop(self, sites, dtype=np.complex128):
        """
        Return a Wilson loop operator. The loop shape is specified by
        the tuples in the sites array

        Parameters
        ----------
        sites : list(tuple(int,int))
            The tuples represent the sites that the loop has to visit.
            The loop is automatically connects the last element to the first one.
        dtype : numpy.dtype, (optional, default = np.complex128)
            dtype of the operator

        Returns
        ----------
        W : quspin.hamiltonian
        """
        # get the loop path

        loop = self.loop(sites)
        # build the operator string (it depends on the orientation of the path)

        op_str = ""
        for prev, next in zip_nearest(sites, periodic=True):
            op_str += make_op_str(next, prev, ("U", "u"))

        op_list = [op_str, [[1.0, *loop]]]
        return qo.hamiltonian([op_list], [], basis=self.qbasis, dtype=dtype, **no_check)

    def string_operator(self, plaquettes, dtype=np.complex128):
        """
        Return a non-local string operator. The string shape is specified
        the plaquettes it has to visits. Each plaquette is denoted with
        the coordinate of its bottom-left corner.

        Parameters
        ----------
        plaquettes : list(tuple(int,int))
            The tuples represent the coordinates of the plaquettes to
            visit. The string can be open-ended, it does not automatically
            loop like the Wilson operator.
        dtype : numpy.dtype, (optional, default = np.complex128)
            dtype of the operator

        Returns
        ----------
        S : quspin.hamiltonian
        """
        # Get the string path
        string = self.string(plaquettes)

        op_str = ""
        for prev, next in zip_nearest(plaquettes, periodic=False):
            op_str += make_op_str(next, prev, ("V", "v"))

        op_list = [op_str, [[1.0, *string]]]
        return qo.hamiltonian([op_list], [], basis=self.qbasis, dtype=dtype, **no_check)

