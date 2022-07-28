# QuSpin-LGT

Exact diagonalization for lattice gauge theories using the [QuSpin](https://github.com/weinbe58/QuSpin) package for python.

## Examples

### Z2 LGT theory
Create an object of the `Z2` class, which represent a Z2 lattice pure
gauge theory (no matter) on `3x3` lattice with periodic boundary condition:
```
from qulgt import Z2
z2_lgt = Z2(size=(3, 3), pbc=(True, True))
```
or you can create it on a `10x2` ladder with periodic condition only along the x
direction:
```
z2_ladder = Z2(size=(3,3), pbc=(True, False))
```

The Z2 class accepts two keyword arguments: `size` and `pbc`. The first one has
to be a tuple of two integers, the second a tuple of two boolean.
The `size` keyword arg specifies the dimension of the lattice (in terms of
sites), while `pbc` whether or not to implements periodic boundary conditions.
The first value of the tuples always refers to the `x` direction, while the
second to the `y` direction.

The class accepts also a third keyword argument `sector`, which can restrict
the Hilbert space to a particular sector between `(0, 0)`,  `(1, 0)`, `(0, 1)` and
`(1, 1)`. The sectors are distinguished by the parity of the number of
non-contractible electric loops along the `x` or `y` direction respectively.
The `0` sign refers to an even number of loops, while `1` to an odd number.
For example:
```
z2_lgt = Z2(size=(3, 3), pbc=(True, True), sector=(0, 0))
```
On the other hand, if we want to explicitly work with the whole gauge invariant
Hilbert space:
```
z2_lgt = Z2(size=(3, 3), pbc=(True, True), sector=None)
```
