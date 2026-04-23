"""This module implements the Lattice base class for the LGT children classes"""

import numpy as np
from itertools import chain, product
from typing import Tuple, List
# from .drawing import LatticeDrawer
from painting import paint_lattice
# from ..utils.iter import zip_nearest

# Type hintings
Site = Tuple[int, int]
Plaquette = Tuple[int, int, int, int]
Star = Tuple[int, int, int, int]

#
def cyclic_perm(lst, n=1):
    return lst[n:] + lst[:n]

def zip_nearest(lst, periodic=True, step=1):
    if periodic:
        return zip(lst, cyclic_perm(lst, n=step))
    else:
        return zip(lst[:-step], lst[step:])


class LatticeError(RuntimeError):
    pass

class SquareLattice:
    def __init__(self, size: Tuple[int, int], pbc: Tuple[bool, bool]):
        """
        Create an instance of the Lattice class for the given size and
        periodic boundary conditions.
        This class is intended to be used as a base class for all the LGT models.

        Parameters
        ----------
        size : tuple(int, int)
            Number of sites along x and y respectively.
        pbc : tuple(bool, bool)
            Periodic boundary conditions along x and y respectively.
        spl : int
            Number of states per link. Basically the dimension of the
            local Hilbert space.
        """
        self.size = size
        # self.spl  = spl
        # TODO: Information about the dimension of the local Hilbert space
        #       should be moved to another class.
        #       The lattice class should care only about the lattice
        self.pbc  = pbc
        # self._set_dtype()
        self.lattice_vectors = [[1, 0], [0, 1]]
        self._create_site_list()
        self._label_links()
        # self._lattice_drawer = LatticeDrawer(self)

    def __repr__(self):
        return f"<SquareLattice: size={self.size}, pbc={self.pbc}>"

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

    def _create_site_list(self):
        """Create list of sites depending on the shape of the lattice"""
        self._sites = {
            (site_inverted[1], site_inverted[0]): index
            for index, site_inverted in enumerate(product(range(self.Ly), range(self.Lx)))
        }

    @property
    def nsites(self) -> int:
        """Returns the total number of sites"""
        return len(self._sites)

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
        return self._sites.get(self._mod_boundary(site), None)

    # def _set_dtype(self):
    #     """Infer the dtype to use through out the class"""
    #     if self.spl**self.nlinks <= 2**32:
    #         self.dtype = np.uint32
    #     elif self.spl**self.nlinks <= 2**64:
    #         self.dtype = np.uint64
    #     else:
    #         raise LatticeError(f"Lattice with size={self.size} and spl={self.spl} is too big")

    def _label_links(self):
        """Labels the links of the lattice"""
        self._link_matrix = np.zeros((self.nsites, self.nsites), dtype=np.int64)

        def neighbor_sites(site):
            x, y = site
            return [(x+e1, y+e2) for e1, e2 in self.lattice_vectors]

        link_index = 1
        for site in self._sites:
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

    def plaquette(self, site: Site, **kwargs) -> Plaquette | None:
        """
        Return the links of a plaquette. The links are oriented
        counterclok-wise starting from the bottom site.

        Parameters
        ----------
        site: tuple[int, int]
            Site coordinates
        type_: int
            Type of plaquette.
            `0` for upward triangle `/\\`, `1` for downward triangle `\\/`
            Default to type 0
        The extra arguments are the same of `link()` function

        Return
        ----------
        tuple[int, int, int] or None
            Returns a 3-element tuple of the link indices of the plaquette.
            If the plaquette does not exist it returns `None`
        """
        x, y = site
        plaq = (
                self.link((x,   y),   (x+1, y),   **kwargs),
                self.link((x+1, y),   (x+1, y+1), **kwargs),
                self.link((x+1, y+1), (x,   y+1), **kwargs),
                self.link((x,   y+1), (x,   y),   **kwargs)
            )
        if None in plaq:
            return None
        return plaq

    def star(self, site: Site, **kwargs) -> Star:
        """
        Return the links of a star. The links are oriented counterclock-wise
        starting from the right link.

        Parameters:
        ----------
        site: tuple[int, int]
            Site coordinates

        Return
        ----------
        ist[int, ...]
            A list of link indices, the size of the list depends on
            the position of the site
        """
        x, y = site
        star = [
            self.link((x, y), (x+1, y),   **kwargs),
            self.link((x, y), (x,   y+1), **kwargs),
            self.link((x, y), (x-1, y), **kwargs),
            self.link((x, y), (x,   y-1),   **kwargs),
        ]
        return [ s for s in star if s is not None ]

    @property
    def sites(self) -> List[Site]:
        return list(self._sites.keys())

    def plaquettes(self, **kwargs) -> List[Plaquette]:
        """
        Return a list of all the plaquettes.
        The indices of the plaquettes start from 0 by default.
        """
        kwargs.setdefault('from_zero', True)
        plaquettes = [
                        self.plaquette(site, **kwargs)
                        for site in self.sites
                    ]
        plaquettes = [ p for p in plaquettes if p is not None ]
        return plaquettes

    def stars(self, **kwargs) -> List[Star]:
        """
        Return an array of all the stars.
        The indices of the stars start from 0.
        """
        kwargs.setdefault('from_zero', True)
        stars = [ self.star(site, **kwargs) for site in self.sites ]
        return stars


    def path(self, site0: Site, site1: Site, **kwargs) -> List[int]:
        """
        Return the indices of links that connect two sites that are on a straigth line

        Parameters
        ----------
        site0, site1 : tuple(int,int)
            the coordinates of the start and end of the path

        Return
        ----------
        path : np.array(int)
            A list of the links that connects site0 and site1.
            The indexing of the links starts from 0.

        Raise:
        ----------
        RuntimeError
                if the sites are not aligned
        """
        x0, y0 = site0
        x1, y1 = site1
        kwargs.setdefault('from_zero', True)
        def step(a, b):
            return +1 if a <= b else -1
        # step = lambda a, b: +1 if a <= b else -1
        if y0 == y1:
            sites = [(x, y0) for x in range(x0, x1+step(x0, x1), step(x0, x1))]
        elif x0 == x1:
            sites = [(x0, y) for y in range(y0, y1+step(y0, y1), step(y0, y1))]
        else:
            raise LatticeError(f"The sites ({x0}, {y0}) and ({x1}, {y1}) are not aligned")
        path_ = [self.link(prev, next, **kwargs)
                    for prev, next in zip_nearest(sites, periodic=False)]
        path_ = np.array(path_)
        return path_

    def dual_path(self, plq0: Site, plq1: Site, **kwargs) -> List[int]:
        """
        Given two plaquettes plq0 and plq1, it returns the links that are cut from
        a dual path that start from plq0 and ends in plq1.

        Parameters
        ----------
        plq0, plq1 : tuple(int,int)
            A plaquette is specified by the coordinates of the bottom-left site.

        Returns
        ----------
        dual_path : np.array(int)
            A list of indices of the links that are cut by the dual path.
            Indices start from 0.

        Raises
        ----------
        Exception
            If the plaquettes are not aligned along a straigth line
        """
        x0, y0 = plq0
        x1, y1 = plq1
        kwargs.setdefault('from_zero', True)
        # +1 for positive direction, -1 for negative directions:
        step = lambda a, b: +1 if a <= b else -1
        # get the correct starting and ending points
        # depending if we are going in the positive or negative direction
        begin = lambda a, b: a+1 if a <= b else a
        end   = lambda a, b: b+1 if a <= b else a

        # is the path horizontal or vertical?
        if y0 == y1:
            pairs = [((x, y0), (x, y0+1)) for x in range(begin(x0, x1), end(x0, x1), step(x0, x1))]
        elif x0 == x1:
            pairs = [((x0, y), (x0+1, y)) for y in range(begin(y0, y1), end(y0, y1), step(y0, y1))]
        else:
            raise RuntimeError(f"The sites ({x0}, {y0}) and ({x1}, {y1}) are not aligned")
        dual_path_ = [self.link(prev, next, **kwargs) for prev, next in pairs]
        dual_path_ = np.array(dual_path_, dtype=self.dtype)
        return dual_path_


    def loop(self, sites: List[Site], **kwargs):
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

    def string(self, plqs: List[Site], **kwargs):
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

    def vlinks(self, y, **kwargs):
        """Returns a list of all the vertical links at a given y"""
        if y >= self.nlinks_y or y < 0:
            raise ValueError("Given y parameter exceeds the lattice dimensions")
        return [self.link((x,y), (x,y+1), **kwargs) for x in range(self.Lx)]

    def hlinks(self, y, **kwargs):
        """Returns a list of all the horizontal links at a given y"""
        if y >= self.Ly or y < 0:
            raise ValueError("Given y parameter exceeds the lattice dimensions")
        bulk = [self.link((x,y), (x+1,y), **kwargs) for x in range(self.Lx-1)]
        if self.pbc_x:
            edges = [
                    self.link((self.Lx-1, y), (0, y),       **kwargs),
                    self.link((self.Lx-1, y), (self.Lx, y), **kwargs)
                    ]
        else:
            edges = []
        return (bulk, edges)

    def __str__(self):
        # ret = repr(self) + '\n\n'
        # ret += self._lattice_drawer.draw_lattice()
        # return ret
        return "<NOT IMPLEMENTED>"

    def draw(self, show_links: bool = True, show_sites: bool = False):
        paint_lattice(self, show_links=show_links, show_sites=show_sites)


