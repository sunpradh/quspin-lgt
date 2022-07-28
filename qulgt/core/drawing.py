import numpy as np
from typing import Dict
from functools import reduce

# TODO: DOCUMENTATION
class LatticeDrawer(object):

    default_char_table = dict( hlink='--', vlink='|', site='+')

    def __init__(self, lattice, char_table: Dict=None):
        self.lattice = lattice
        self.Lx      = self.lattice.Lx
        self.pbc_x   = self.lattice.pbc_x
        self.char_table = char_table if char_table is not None \
                            else self.default_char_table

    def draw_row_links(self, y: int) -> str:
        """Draw the horizontal links of a given row"""
        hlink = self.char_table['hlink']
        site  = self.char_table['site']
        blank  = ' ' * len(hlink)
        bigblank = blank * 2 + '  '
        bulk, edges = self.lattice.hlinks(y, from_zero=True)
        left  = blank + f'{edges[0]:>2d}' + hlink if self.pbc_x else bigblank
        right = hlink + f'{edges[1]:<2d}' + blank if self.pbc_x else bigblank
        left  = left  + site + hlink
        right = hlink + site + right
        inner = (hlink + site + hlink).join(f'{link:2d}' for link in bulk)
        return left + inner + right + '\n'

    def draw_vert_empty(self) -> str:
        """Draw empty vertical spaces"""
        blank = ' ' * (2 * len(self.char_table['hlink']) + 2)
        vlink = self.char_table['vlink']
        return vlink.join(blank for _ in range(self.Lx+1)) + '\n'

    def draw_vert_links(self, y: int) -> str:
        """Draw the vertical links at a given y"""
        links = self.lattice.vlinks(y, from_zero=True)
        blank = ' ' * (2 * len(self.char_table['hlink']) + 2)
        return blank + blank[1:].join(f'{link:<2d}' for link in links) + '\n'

    def draw_lattice(self) -> str:
        """Draw the lattice"""
        picture = ''
        for y in range(self.lattice.Ly-1, 0, -1):
            picture += self.draw_row_links(y) \
                 + self.draw_vert_empty() \
                 + self.draw_vert_links(y-1) \
                 + self.draw_vert_empty()
        picture += self.draw_row_links(0)

        if self.lattice.pbc_y:
            picture = self.draw_vert_links(self.lattice.Ly-1) \
                + self.draw_vert_empty() \
                + picture \
                + self.draw_vert_empty() \
                + self.draw_vert_links(self.lattice.Ly-1)
        return picture


class StateDrawer():
    def __init__(self, lattice):
        self.lattice = lattice
        self.spl     = lattice.spl
        self.N       = lattice.nlinks
        self._offsetlength = 4
        self.set_default_char_table()

    def set_char_table(self, int_repr, site_sym, hlinks, vlinks):
        self.char_table = dict(
                int=int_repr,
                site=site_sym,
                hlinks=hlinks,
                vlinks=vlinks
                )

    def update_char_table(self, **kwargs):
        for key, value in kwargs.items():
            self.char_table.update([(key, value)])

    def set_default_char_table(self):
        self.set_char_table(
                int_repr=None,
                site_sym='·',
                hlinks=[' '] + [f'{s:1d}' for s in range(1, self.spl)],
                vlinks=[' '] + [f'{s:1d}' for s in range(1, self.spl)]
                )

    @property
    def offset(self):
        return ' ' * self._offsetlength

    @offset.setter
    def offset(self, os):
        self._offsetlength = os

    @property
    def site(self):
        return self.char_table['site']

    def hlink(self, occ):
        return self.char_table['hlinks'][occ]

    def vlink(self, occ):
        return self.char_table['vlinks'][occ]

    def occ(self, state, pos):
        return np.uint((state // self.spl**pos) % self.spl)

    def link_occ(self, state, site0, site1):
        return self.occ(state, self.lattice.link(site0, site1, flip=True))

    def int_repr(self, state):
        irepr = reduce(
            lambda a, b: b+a,
            [str(self.occ(state, pos)) for pos in range(self.N)]
        )
        if self.char_table['int'] is not None:
            for k, v in self.char_table['int'].items():
                irepr = repr.replace(k, v)
        return irepr

    def draw_vlinks(self, state, y):
        vlinks = self.lattice.vlinks(y, flip=True)
        ret = self.offset \
            + ' ' \
            + ' '.join([self.vlink(self.occ(state, link)) for link in vlinks])
        return ret

    def draw_hlinks(self, state, y):
        bulk, edges = self.lattice.hlinks(y, flip=True)
        ret = self.offset + self.hlink(self.occ(state, edges[0])) + self.site
        ret += self.site.join([self.hlink(self.occ(state, link)) for link in bulk])
        ret += self.site + self.hlink(self.occ(state, edges[1]))
        return ret

    def draw_state(self, state):
        Ly = self.lattice.Ly
        ret = ''
        for y in range(Ly-1, 0, -1):
            ret += self.draw_hlinks(state, y)   + '\n'
            ret += self.draw_vlinks(state, y-1) + '\n'
        ret += self.draw_hlinks(state, 0)
        if self.lattice.pbc_y:
            ret = self.draw_vlinks(state, Ly-1) \
                + '\n' + ret + '\n' \
                + self.draw_vlinks(state, Ly-1) + '\n'
        return ret


