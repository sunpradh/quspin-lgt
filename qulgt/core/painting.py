# import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
# from .trilattice import TriangularLattice

def paint_lattice(lattice, transform: transforms.Transform | None = None):
    sites = lattice.sites
    fig, ax = plt.subplots()
    if transform is None:
        tr = ax.transData
    else:
        tr = transform + ax.transData
    xcoords = [ s[0] for s in sites ]
    ycoords = [ s[1] for s in sites ]
    ax.scatter(xcoords, ycoords, zorder=10, transform=tr)
    for s0 in sites:
        ax.text(s0[0], s0[1]-0.15, f"({s0[0]}, {s0[1]})", transform=tr,
                ha="center",
                va="bottom",
                backgroundcolor="#ffffff99",
                color="blue",
                zorder=15
                )
        for s1 in sites:
            link = lattice.link(s0, s1)
            if link:
                ax.plot([s0[0], s1[0]], [s0[1], s1[1]], "k", transform=tr, zorder=1)
                midpoint_x = (s0[0] + s1[0]) / 2
                midpoint_y = (s0[1] + s1[1]) / 2
                ax.text(midpoint_x, midpoint_y, f"{link}", transform=tr,
                    va="center",
                    backgroundcolor="#ffffff99",
                    zorder=3
                    )
