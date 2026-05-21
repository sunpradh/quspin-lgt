"""
This module implements the Zn lattice gauge theory base class
"""
import numpy as np
from itertools import product, groupby
import quspin.operators as qo

from ..core.gauge_theory import GaugeTheoryBase, ModelError
from ..core.mkstates     import StateGenerator
from ..utils.iter        import zip_nearest

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
        Same arguments of `GaugeTheoryBase`

        Additional parameters
        ----------
        sector: tuple[int, int] or string (default: 'all')
            Select the sector of the model.
            In the case of PBC, will `0, ..., spl-1` available sectors for each
            periodic direction, where `spl` is the number of states per link.
            Otherwise, only one sector for each non-periodic direction is available.
            Each sector is identified with a pair of integers `(n_x, n_y)`.
            Otherwise, the keyword 'all' can be used to select all of them and
            build the full physical Hilber space.
        """
        super().__init__(*args, **kwargs)


    def __repr__(self):
        return f'<ZnBase on {self.lattice_repr()}, {self.Ns} states, sector {self.sector}>'


    def _mkstates(self, sector: tuple[int, int] | str = "all"):
        """
        Make all the states for the basis.

        If a sector is specified with a pair of int then only the states
        in that sector are constructed.
        """
        states_generator = StateGenerator(lattice=self.lattice, spl=self.spl, dtype=self.dtype)
        if sector != 'all':
            if sector in self._avail_sectors():
                self.sector = sector
                states = states_generator(self._get_sector_vacuum(sector))
            else:
                raise ModelError(f"The sector {sector} is not valid")
        else:
            self.sector = 'all'
            states = [
                states_generator(self._get_sector_vacuum(sect))
                for sect in self._avail_sectors()
            ]
            states = np.array(states, dtype=self.dtype).ravel()
        return states


    def _get_sector_vacuum(self, sector: tuple[int, int]):
        """Vacuum state for a given sector"""
        if sector not in self._avail_sectors():
            raise RuntimeError(f"Specified sector {sector} is not valid")
        vacuum = self.dtype(0)
        print(f"[_get_sector_vacuum] dtype = {self.dtype}")
        if sector == (0, 0):
            return vacuum
        sector_x, sector_y = sector
        if self.lattice.pbc_x:
            loop_x = self.lattice.line((0, 0), (self.lattice.Lx, 0), from_zero=True, flip=True)
            print(f"[_get_sector_vacuum] Creating loops along x, loop_x = {loop_x}")
            for _ in range(sector_x):
                for l in loop_x:
                    vacuum += self.dtype(self.spl ** int(l))
        if self.lattice.pbc_y:
            print(f"[_get_sector_vacuum] Creating loops along y, loop_y = {loop_x}")
            loop_y = self.lattice.line((0, 0), (0, self.lattice.Ly), from_zero=True, flip=True)
            for _ in range(sector_y):
                for l in loop_y:
                    vacuum += self.dtype(self.spl ** int(l))
        return vacuum


    def _avail_sectors(self):
        x_sectors = range(self.spl if self.lattice.pbc_x else 1)
        y_sectors = range(self.spl if self.lattice.pbc_y else 1)
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
        plqs = self.lattice.plaquettes(from_zero=True)
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
        link_array = np.zeros((self.lattice.nlinks, 2))
        link_array[:, 0] = coupling
        link_array[:, 1] = np.arange(self.lattice.nlinks)
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

    def coupling_plaquettes_gen(self, coupling=0.0, include_type: bool = False):
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
        if not include_type:
            plqs = self.lattice.plaquettes(from_zero=True)
            nplaqlinks = len(plqs[0])
            plq_arr = np.zeros((len(plqs), nplaqlinks+1))
            plq_arr[:, 1:] = np.array(plqs)
            plq_arr[:, 0 ] = coupling
            plq_arr = plq_arr.tolist()
            return [([plq_arr[i] for i in range(len(plqs))], None)]
        else:
            plqs = self.lattice.plaquettes(from_zero=True,include_type=include_type)
            nplaqlinks = len(plqs[0][0])
            which = [plaq[1] for plaq in plqs]
            plqs = [plaq[0] for plaq in plqs]
            plq_arr = np.zeros((len(plqs), nplaqlinks+1))
            plq_arr[:, 1:] = np.array(plqs)
            plq_arr[:, 0 ] = coupling
            plq_arr = plq_arr.tolist()
            list_array = [(plq_arr[i], which[i]) for i in range(len(plqs))]
            data = sorted(list_array, key=lambda x: x[1])

            list_array = [(np.stack([v[0] for v in group]), key) 
                    for key, group in groupby(data, key=lambda x: x[1])]
            return [[elem[0].tolist(), elem[1]] for elem in list_array] 
    
    def mk_plqs_str_tri_type_0(self, conj: bool = False):
        if not conj:
            return "UUu"
        else:
            return "uuU"
    
    def mk_plqs_str_tri_type_1(self, conj: bool = False):
        if not conj:
            return "Uuu"
        else:
            return "uUU"
    
    def mk_plqs_str_square(self, conj: bool = False):
        if not conj:
            return "UUuu"
        else:
            return "uuUU"

    def mk_plqs_str(self, conj: bool = False, which = None):
        if which is None:
            return self.mk_plqs_str_square(conj=conj)
        elif which == 0:
            return self.mk_plqs_str_tri_type_0(conj=conj)
        elif which == 1:
            return self.mk_plqs_str_tri_type_1(conj=conj)
    
    def mk_plqs_list(self, coupling=0.0, include_type: bool = False):
        """
        Return a list of the plaquettes with the operator string 
        related to the plaquettes, and the conjugate version. In case the
        type is False, we fall to the simple case of square plaquettes.
        If include_type is True, we have different types of plaquettes
        which have a different string operator to represent the quantum operation.

        Parameters
        ----------
        coupling : float (default: 0.0)
            coupling on the plaquette term
        include_type : bool (default: False)
            we can have different operator plaquette terms according to the lattice
            geometry. If False it gives the links of a square plaquette, else a triangular
            lattice has two types of plaquettes i.e. either `/\\` (type 0) or `\\/` (type 1)
        """
        plq_arr = self.coupling_plaquettes_gen(coupling=coupling, include_type=include_type)
        plq_conj_arr = self.coupling_plaquettes_gen(coupling=np.conj(coupling), include_type=include_type)
        plaq_list = [[self.mk_plqs_str(conj=False, which=which), coup_plqs] for (coup_plqs, which) in plq_arr]
        plaq_conj_list = [[self.mk_plqs_str(conj=True, which=which), coup_plqs] for (coup_plqs, which) in plq_conj_arr]
        return plaq_list + plaq_conj_list
    
    def hamiltonian_gen(self, plq, elec, dtype=np.complex128):
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
        coupling_links      = self.coupling_links(elec)
        coupling_links_conj = self.coupling_links(np.conj(elec))
        links_list = [
            ["V",    coupling_links],     # V
            ["v",    coupling_links_conj] # Vdag
        ]
        if len(self.lattice.lattice_vectors) == 3:
            include_type = True
        else:
            include_type = False
        plqs_list = self.mk_plqs_list(coupling=plq, include_type=include_type)
        slist = links_list + plqs_list

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
            self._quantum_operator = qo.quantum_operator(input_dict, N=self.lattice.nlinks, basis=self.qbasis, dtype=dtype, **no_check)

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

        loop = self.lattice.loop(sites)
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
        string = self.lattice.string(plaquettes)

        op_str = ""
        for prev, next in zip_nearest(plaquettes, periodic=False):
            op_str += make_op_str(next, prev, ("V", "v"))

        op_list = [op_str, [[1.0, *string]]]
        return qo.hamiltonian([op_list], [], basis=self.qbasis, dtype=dtype, **no_check)

