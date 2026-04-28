"""This module implements the base class for all lattices"""


import numpy as np
from abc import ABC, abstractmethod
from itertools import chain
from typing import TypeVar

from qulgt.utils.iter import zip_nearest
from qulgt.core.painting import paint_lattice

# Type hinting
Site = tuple[int, int]
Plaquette = tuple[int, ...]
Star = tuple[int, ...]


class LatticeError(Exception):
    pass


class BaseLattice(ABC):
    def __init__(self, size: tuple[int, int], pbc: tuple[bool, bool], vectors: list[list[int]]):
        self.size =  size
        self.pbc = pbc
        self.lattice_vectors = self._validate_lattice_vectors(vectors)
        self._sites_dict = self._create_site_dict()
        self._label_links()


    def __repr__(self):
        return f"<BaseLattice: size={self.size}, pbc={self.pbc}, vectors={self.lattice_vectors}>"


    def _validate_lattice_vectors(self, vector_list: list[list[int]]):
        lattice_vecs = []
        for obj in vector_list:
            vector = np.array(obj)
            if vector.dtype != int:
                raise ValueError(f"The lattice vectors must be integer valued. The following passed vector is not: {vector}")
            if vector.shape != (2,):
                raise ValueError(f"The lattice vectors must have shape (2,). The following passed vector is not: {vector}")
            lattice_vecs.append(vector)
        return lattice_vecs


    @abstractmethod
    def _create_site_dict(self):
        """Create dictionary where the keys are the site coordinates and the values the site index"""
        pass

    @abstractmethod
    def plaquette(self, site: Site):
        """Returns the links of a plaquette"""
        pass

    @abstractmethod
    def star(self, site: Site):
        """Return the links attached to a site"""
        pass

    # TODO no idea on how to implement this generically
    @abstractmethod
    def dual_line(self, plq0: Site, plq1: Site, **kwargs) -> list[int]:
        pass


    @property
    def Lx(self) -> int:
        """Returns the horizontal length"""
        return self.size[0]

    @property
    def Ly(self) -> int:
        """Returns the vertical length"""
        return self.size[1]

    @property
    def pbc_x(self) -> bool:
        """Returns if x-axis is period"""
        return self.pbc[0]

    @property
    def pbc_y(self) -> bool:
        """Returns if y-axis is period"""
        return self.pbc[1]

    @property
    def nsites(self) -> int:
        """Returns the total number of sites"""
        return len(self._sites_dict)

    @property
    def sites(self) -> list[Site]:
        return list(self._sites_dict.keys())


    def _mod_boundary(self, site: Site) -> Site:
        """
        Returns the modulo of the site coordinates in the presence
        of periodic boundary conditions
        """
        x, y = site
        x = x % self.Lx if self.pbc_x else x
        y = y % self.Ly if self.pbc_y else y
        return (x, y)


    def site_index(self, site: Site) -> int | None:
        """Return the integer index of a site, given its coordinates

        Parameters
        ----------
        site: tuple[int, int]
            Site coordinates

        Return
        ----------
        int or None
            Returns the site index if it exist, otherwise `None`
        """
        return self._sites_dict.get(self._mod_boundary(site), None)


    def _label_links(self):
        """Labels the links of the lattice"""
        self._link_matrix = np.zeros((self.nsites, self.nsites), dtype=np.int64)

        def neighbor_sites(site):
            x, y = site
            return [(x+ex, y+ey) for ex, ey in self.lattice_vectors]

        link_index = 1
        for site in self.sites:
            x, y = site
            index0 = self.site_index(site)
            for next_site in neighbor_sites(site): # horizontal, vertical and then diagonal
                index1 = self.site_index(next_site)
                if index1 is None:
                    continue
                self._link_matrix[index0, index1] = link_index
                self._link_matrix[index1, index0] = link_index
                link_index = link_index + 1
        self.nlinks = link_index - 1


    def link(self, site0: Site, site1: Site, flip=False, from_zero=False) -> int | None:
        """Return the link index between two sites

        Parameters
        ----------
        site0 : tuple[int,int]
        site1 : tuple[int,int]
        flip : bool (default=False)
            whether to flip the integer repr of the link indices.
        from_zero : bool (default=False)
            whether to start the indexing from zero or not.
            May cause problems.

        the coordinates are taken mod Lx (for x_i) or mod Ly (for y_i)
        The value 0 (with from_zero=False) is used to indicate an absent link.

        Returns
        ----------
        int or None
            Return an the link index if it exist, `None` otherwise
        """
        index0 = self.site_index(site0)
        index1 = self.site_index(site1)
        if (index0 is None) or (index1 is None):
            return None
        link_index = self._link_matrix[index0, index1]
        if link_index == 0:
            return None
        if from_zero:
            link_index = link_index - 1
        if flip:
            link_index = self.nlinks - link_index - 1*from_zero
        return int(link_index)


    def plaquettes(self, **kwargs) -> list[Plaquette]:
        """
        Return a list of all the plaquettes.
        The indices of the plaquettes start from 0 by default.
        """
        kwargs.setdefault('from_zero', True)
        plaquettes = [ self.plaquette(site, **kwargs) for site in self.sites ]
        plaquettes = [ p for p in plaquettes if p is not None ]
        return plaquettes


    def stars(self, **kwargs) -> list[Star]:
        """
        Return an array of all the stars.
        The indices of the stars start from 0.
        """
        kwargs.setdefault('from_zero', True)
        stars = [ self.star(site, **kwargs) for site in self.sites ]
        return stars


    def line(self, site0: Site, site1: Site, **kwargs) -> list[int] | None:
        """
        Return the indices of the links that lies on the straigth line connecting two sites

        Parameters
        ----------
        site0, site1 : tuple[int, int]
            the coordinates of the start and end of the path

        Return
        ----------
        path : list[int] or None
            A list of the links that connects site0 and site1.
            The indexing of the links starts from 0.
            Return `None` if no links are found

        Raise:
        ----------
        LatticeError
            If the sites do not exists or are not aligned
        """
        if (self.site_index(site0) is None) or (self.site_index(site1) is None):
            raise LatticeError(f"Invalid sites {site0} and {site1}")
        x0, y0 = site0
        x1, y1 = site1
        diff = np.array([x1 - x0, y1 - y0])
        sites = []
        are_aligned = False
        for lat_vec in self.lattice_vectors:
            ex, ey = lat_vec
            # Check if `diff` and `lat_vec` are aligned while avoiding floating point operations.
            # (we want to preserve the integer coordinates)
            #  -> if `diff` and `lat_vec` are aligned
            #  -> then `diff` is orthogonal to 90-degrees rotated `lat_vec` (-ey, ex)
            check = - diff[0] * ey + diff[1] * ex # scalar product
            if check == 0:
                are_aligned = True
                nsteps = diff[0] // ex if ex != 0 else diff[1] // ey
                sign = nsteps // abs(nsteps)
                sites = [(x0 + n*ex, y0 + n*ey) for n in range(0, nsteps+sign, sign)]
                break
        if not are_aligned:
            raise LatticeError(f"The sites {site0} and {site1} are not aligned")
        if len(sites) == 0:
            return None
        path_ = [self.link(prev, next, **kwargs) for prev, next in zip_nearest(sites, periodic=False)]
        return path_


    def path(self, *args, **kwargs) -> list[int]:
        print("DEPRECATED use `.line()` instead of `.path()`")
        return self.line(*args, **kwargs)


    def dual_path(self, *args, **kwargs) -> list[int]:
        print("DEPRECATED use `.dual_line()` instead of `.dual_path()`")
        return self.dual_line(*args, **kwargs)


    def loop(self, sites: list[Site], **kwargs) -> np.array(int):
        """
        Return a list of indices of a loop given a list of the site to visits.
        The sites are specified by tuple of two integers.

        Parameters
        ----------
        sites : list(tuple(int,int))

        Return
        ----------
        loop : np.array(int)
            A list of the indices of all the link visited
        """
        loop_indx = [ self.path(prev, next, **kwargs) for prev, next in zip_nearest(sites, periodic=True) ]
        loop_indx = list(chain.from_iterable(loop_indx))
        return np.array(loop_indx, dtype=self.dtype)


    def string(self, plqs: list[Site], **kwargs):
        """
        Return a list of indices of a loop given a list of the site to visit.

        Parameters
        ----------
        plqs : list(tuple(int,int))

        Return
        ----------
        plaquettes : np.array(int)
            A list of the indices of all the link visited
        """
        string_indx = [ self.dual_path(prev, next, **kwargs)
                            for prev, next in zip_nearest(plqs, periodic=False)]
        string_indx = list(chain.from_iterable(string_indx))
        return np.array(string_indx, dtype=self.dtype)


    def draw(self, show_links: bool = True, show_sites: bool = False):
        """
        Paint a picture of the lattice

        Parameters
        ----------
        show_links: bool [default True]
            Display the indices of the links
        show_sites: bool [default False]
            Display the coordinates of the sites
        """
        paint_lattice(self, show_links=show_links, show_sites=show_sites)


# Type for all subclasses of BaseLattice
LatticeT = TypeVar('LatticeT', bound=BaseLattice)
