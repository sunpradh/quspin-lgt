"""This module implements the square lattice"""

from itertools import product
from qulgt.core.base_lattice import BaseLattice, Site, Plaquette, Star, LatticeError


class SquareLattice(BaseLattice):
    def __init__(self, size: tuple[int, int], pbc: tuple[bool]):
        """
        Create a square lattice given size, shape and boundary conditions

        Parameters
        ----------
        size : tuple[int, int]
            Size of the horizontal and vertical sides, respectively.
        pbc : tuple[bool, bool] (default: (False, False))
            Whether to have periodic boundary conditions along x and y respectively.
            PBCs are available only for the parallelogram shape.
        """
        super().__init__(size=size, pbc=pbc, vectors=[[1, 0], [0, 1]])


    def __repr__(self):
        return f"<SquareLattice: size={self.size}, pbc={self.pbc}>"


    def _create_site_dict(self):
        """Create list of sites depending on the shape of the lattice"""
        # invert the site components in order to have the correct ordering (first horizontal then vertical)
        # when doing the cartesian product
        return {
            (x, y): index
            for index, (y, x) in enumerate(product(range(self.Ly), range(self.Lx)))
        }


    def plaquette(self, site: Site, **kwargs) -> Plaquette | None:
        """
        Return the links of a plaquette. The links are oriented
        counterclok-wise starting from the bottom site.

        Parameters
        ----------
        site: tuple[int, int]
            Site coordinates
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


    def dual_line(self, plq0: Site, plq1: Site, **kwargs) -> list[int]:
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
        # +1 for positive direction, -1 for negative directions:
        step = lambda a, b: +1 if a <= b else -1
        # get the correct starting and ending points
        # depending if we are going in the positive or negative direction
        begin = lambda a, b: a+1 if a <= b else a
        end   = lambda a, b: b+1 if a <= b else b

        # is the path horizontal or vertical?
        if y0 == y1:
            pairs = [((x, y0), (x, y0+1)) for x in range(begin(x0, x1), end(x0, x1), step(x0, x1))]
        elif x0 == x1:
            pairs = [((x0, y), (x0+1, y)) for y in range(begin(y0, y1), end(y0, y1), step(y0, y1))]
        else:
            raise LatticeError(f"The sites ({x0}, {y0}) and ({x1}, {y1}) are not aligned")
        dual_path_ = [self.link(prev, next, **kwargs) for prev, next in pairs]
        return dual_path_
