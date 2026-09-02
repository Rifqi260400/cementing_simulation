"""Reduced-order in-pipe displacement solver.

A reconstruction of the in-pipe displacement model of Dai et al. (2023),
"Modeling displacement flow inside a full-length casing string for well
cementing", Petroleum Research 9, 1-16.

This is *not* a CFD solver.  It couples an analytical 1D axial velocity
profile, solved per depth station, to a 3D scalar transport of fluid volume
fractions advected by that axial velocity only.
"""

__version__ = "0.1.0"
