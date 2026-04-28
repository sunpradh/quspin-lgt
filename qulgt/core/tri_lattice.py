"""This module implements the triangular lattice"""

import math
from matplotlib import transforms
from qulgt.core.base_lattice import BaseLattice, Site, Plaquette, Star, LatticeError
from qulgt.core.painting import paint_lattice


class TriangularLattice(BaseLattice):
    def __init__(self, shape: str, size: tuple[int, int], pbc: tuple[bool, bool] = (False, False)):
        self._check_boundary_conditions(shape, pbc)
        self.lattice_shape = shape
        super().__init__(size=size, pbc=pbc, vectors=[[1, 0], [0, 1], [-1, 1]])


    def __repr__(self):
        return f"<TriangularLattice: shape={self.shape}, size={self.size}, pbc={self.pbc}>"


    def _check_boundary_conditions(self, shape: str, pbc: tuple[bool, bool]):
        if pbc[0] == True or pbc[1] == True:
            if shape != "parallelogram" and shape != "par":
                raise LatticeError(f"Periodic boundary conditions are valid only for `parallelogram` shape.\n\
                    The shape `{shape}` has been passed")


    def _create_site_dict(self):
        length_x, length_y = self.size
        match self.lattice_shape:
            case "tri" | "triangle":
                site_list = [ (x, y) for y in range(length_y) for x in range(length_x - y)]
            case "par" | "parallelogram":
                site_list = [ (x, y) for y in range(length_y) for x in range(length_x)]
            case "hex" | "hexagon":
                mid_length = length_y // 2
                lst1 = [ (x, y) for y in range(mid_length) for x in range(-y, length_x)]
                lst2 = [ (x, y) for y in range(mid_length, length_y) for x in range(-mid_length, length_x - (y - mid_length)) ]
                site_list = lst1 + lst2
            case _:
                raise LatticeError(f"Shape `{self.lattice_shape}` not recognized")
        return {site: index for index, site in enumerate(site_list)}


    def plaquette(self, site: Site, which: int = 0, **kwargs) -> Plaquette | None:
        """
        Return the links of a plaquette. The links are oriented
        counterclok-wise starting from the bottom site.

        Parameters
        ----------
        site: tuple[int, int]
            Site coordinates
        which: int [default 0]
            Which type of plaquette.
            `0` for upward triangle `/\\`, `1` for downward triangle `\\/`
            In the case of upward triangle: `site` referes to the lower left corner
            In the case of downward triangle: `site` referes to the bottom corner
        The extra arguments are the same of `link()` function

        Return
        ----------
        tuple[int, int, int] or None
            Returns a 3-element tuple of the link indices of the plaquette.
            If the plaquette does not exist it returns `None`
        """
        x, y = site
        if which == 0:
            plaq = (
                    self.link((x,   y), (x+1, y), **kwargs),
                    self.link((x+1, y), (x, y+1), **kwargs),
                    self.link((x, y+1), (x, y),   **kwargs)
                )
        elif which == 1:
            plaq = (
                    self.link((x,   y),   (x, y+1), **kwargs),
                    self.link((x,   y+1), (x-1, y+1), **kwargs),
                    self.link((x-1, y+1), (x, y),   **kwargs)
                )
        else:
            raise LatticeError(f"Unrecognized type of plaquette: site={site}, which={which}")
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
            self.link((x, y), (x-1, y+1), **kwargs),
            self.link((x, y), (x-1, y),   **kwargs),
            self.link((x, y), (x, y-1),   **kwargs),
            self.link((x, y), (x+1, y-1), **kwargs)
        ]
        return [ s for s in star if s is not None ]


    def dual_line(self, plq0: tuple[int, ...], plq1: tuple[int, ...], **kwargs) -> list[int]:
        """
        Given two plaquettes plq0 and plq1, it returns the links that are cut from
        a dual path that start from plq0 and ends in plq1.

        Parameters
        ----------
        plq0, plq1 : tuple[int, int, int]
            A plaquette is specified by the sites coordinates and the type (either 0 or 1)
            Site coordinates and type follow the same specification of `.plaquette()`

        Returns
        ----------
        dual_path : list[int]
            A list of indices of the links that are cut by the dual path.
            Indices start from 0.

        Raises
        ----------
        Exception
            If the plaquettes are not aligned along a straigth line
        """
        # TODO IMPLEMENTATION
        raise RuntimeError("This method is still not implemented")
        x0, y0, type0 = plq0
        x1, y1, type1 = plq1
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
        return dual_path_


    def draw(self, show_links: bool = True, show_sites: bool = False, square: bool = False):
        """
        Paint a picture of the lattice

        Parameters
        ----------
        show_links: bool [default True]
            Display the indices of the links
        show_sites: bool [default False]
            Display the coordinates of the sites
        square: bool [default False]
            Draw the lattice on a square grid
        """
        if not square:
            transf = transforms.Affine2D().from_values(1, 0, 1/2, math.sqrt(3)/2, 0, 0)
            paint_lattice(self, transform=transf, show_links=show_links, show_sites=show_sites)
        else:
            paint_lattice(self, show_links=show_links, show_sites=show_sites)
