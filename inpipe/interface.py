"""Module 5 - interface reconstruction (Phase 2).

Deliberately empty in Phase 1.  The paper states that a sharp interface is
"maintained by applying axial interface reconstruction to suppress numerical
diffusion because of large axial grid size" (Appendix A.2) but never describes
the scheme, so there is nothing here to reconstruct faithfully.  Phase 1
establishes the first-order upwind baseline and measures its smearing; Phase 2
is the comparative study of donor-acceptor and THINC against that baseline.

To add a scheme, register a face-value function with the same signature as
:func:`inpipe.transport.upwind_faces` in
:data:`inpipe.transport.FACE_SCHEMES`; the solver takes it from
``NumericsConfig.face_scheme`` with no other change.
"""

from __future__ import annotations

__all__: list[str] = []
