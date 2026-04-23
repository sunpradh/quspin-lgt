import matplotlib.pyplot as plt
import matplotlib.transforms as transforms


def paint_lattice(lattice,
                  transform: transforms.Transform | None = None,
                  show_links: bool = True,
                  show_sites: bool = False
                  ):
    sites = lattice.sites
    fig, ax = plt.subplots()

    # Apply transformations if given
    if transform is None:
        tr = ax.transData
    else:
        tr = transform + ax.transData

    # Draw sites
    xcoords = [ s[0] for s in sites ]
    ycoords = [ s[1] for s in sites ]
    ax.scatter(xcoords, ycoords, zorder=15, transform=tr)

    # Draw links
    for k, site in enumerate(sites):
        x0, y0 = site

        if show_sites: # site labels
            ax.text(x0, y0, f"{site}", transform=tr,
                    ha="center",
                    va="bottom",
                    backgroundcolor="#ffffffcc",
                    color="blue",
                    zorder=10
                    )

        for next_site in sites[k+1:]:
            x1, y1 = next_site
            link = lattice.link(site, next_site)
            if link is None:
                continue
            # Default case:
            xs, ys = (x0, x1), (y0, y1)
            linestyle = "-"

            # Check boundary in case of PBC
            # Boundary-crossing links are shown as dotted lines
            if lattice.pbc_x or lattice.pbc_y:
                for vec in lattice.lattice_vectors:
                    ex, ey = vec
                    # `next_site` connected to `site` by a lattice vector?
                    if lattice.site_index((x1+ex, y1+ey)) == lattice.site_index(site):
                        xs, ys = (x1, x1+ex), (y1, y1+ey)
                        linestyle = ":"
                    # `next_site` connected to `site` while crossing the boundary?
                    if (lattice._mod_boundary((x0 + ex, y0 + ey)) != (x0 + ex, y0 + ey))\
                        and (lattice.site_index((x0+ex, y0+ey)) == lattice.site_index(next_site)):
                        xs, ys = (x1, x1-ex), (y1, y1-ey)
                        linestyle = ":"

            # Plot the links
            ax.plot(xs, ys, "k", linestyle=linestyle, transform=tr, zorder=1)
            if show_links:
                text_x, text_y = (xs[0] + xs[1]) / 2, (ys[0] + ys[1]) / 2
                ax.text(text_x, text_y, f"{link}", transform=tr,
                        ha="center",
                        va="center",
                        backgroundcolor="#ffffffcc",
                        zorder=3
                    )
