"""Case files - fluid properties and geometry as data, not code.

Everything that a study varies lives in a JSON file rather than in a script, so
properties can be changed without editing Python::

    {
      "name": "Tao et al. 2025",
      "fluids": {
        "displaced":  {"name": "drilling fluid", "rho": 998,  "tau0": 0.0, "k": 1e-3, "n": 1.0},
        "displacing": {"name": "cement slurry",  "rho": 1200, "tau0": 1.4, "k": 0.6,  "n": 0.4}
      },
      "geometry":  {"length": 1.0, "casing_id": 0.16, "casing_od": 0.20,
                    "hole_diameter": 0.40},
      "flow":      {"inlet_velocity": 0.5},
      "interface": {"surface_tension": 0.07},
      "rheology":  {"regularisation_shear_rate": 5.5, "normalise_consistency": false}
    }

Only ``fluids`` is required.  Unknown keys are rejected rather than ignored, so
a typo in a property name fails loudly instead of silently leaving the default
in place - which in a validation study would be the worst kind of quiet error.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import InterfaceConfig
from .fluid import Fluid

__all__ = ["CaseSpec", "load_case", "save_case", "DEFAULT_CASE"]

_FLUID_KEYS = {"name", "rho", "tau0", "k", "n"}
_GEOMETRY_KEYS = {"length", "casing_id", "casing_od", "hole_diameter", "top_depth",
                  "rat_hole_length", "bit_diameter"}
_FLOW_KEYS = {"inlet_velocity", "flow_rate", "excess"}
_INTERFACE_KEYS = {"surface_tension"}
_RHEOLOGY_KEYS = {"regularisation_shear_rate", "normalise_consistency"}
_SECTIONS = {"name", "notes", "fluids", "geometry", "flow", "interface", "rheology"}


@dataclass
class CaseSpec:
    """A complete, editable description of a displacement case."""

    displaced: Fluid
    displacing: Fluid
    name: str = "case"
    notes: str = ""
    geometry: dict = field(default_factory=dict)
    flow: dict = field(default_factory=dict)
    interface: InterfaceConfig = field(default_factory=InterfaceConfig)
    regularisation_shear_rate: float | None = None
    normalise_consistency: bool = False

    def summary(self) -> str:
        lines = [f"case: {self.name}"]
        if self.notes:
            lines.append(f"  {self.notes}")
        for role, fl in (("displaced", self.displaced), ("displacing", self.displacing)):
            kind = ("Newtonian" if fl.is_newtonian
                    else "power-law" if fl.tau0 == 0.0 else "Herschel-Bulkley")
            lines.append(
                f"  {role:11s}: {fl.name!r} rho {fl.rho:.1f} kg/m3, {kind}, "
                f"tau0 {fl.tau0:g} Pa, k {fl.k:g} Pa.s^n, n {fl.n:g}"
            )
        if self.geometry:
            lines.append("  geometry   : "
                         + ", ".join(f"{k} {v:g}" for k, v in sorted(self.geometry.items())))
        if self.flow:
            lines.append("  flow       : "
                         + ", ".join(f"{k} {v:g}" for k, v in sorted(self.flow.items())))
        lines.append(f"  interface  : surface tension {self.interface.surface_tension:g} N/m "
                     "(reported, not modelled)")
        if self.regularisation_shear_rate is None:
            lines.append("  rheology   : exact Herschel-Bulkley, rigid plug")
        else:
            lines.append(
                f"  rheology   : Fluent-style regularisation at "
                f"{self.regularisation_shear_rate:g} 1/s, no plug"
                + ("  [consistency normalised by gc]" if self.normalise_consistency else "")
            )
        return "\n".join(lines)


def _check_keys(section: str, data: dict, allowed: set):
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(
            f"unknown key(s) {sorted(unknown)} in section {section!r}; "
            f"allowed: {sorted(allowed)}"
        )


def _fluid_from(role: str, data: dict) -> Fluid:
    _check_keys(f"fluids.{role}", data, _FLUID_KEYS)
    missing = {"rho", "tau0", "k", "n"} - set(data)
    if missing:
        raise ValueError(f"fluids.{role} is missing {sorted(missing)}")
    return Fluid(name=data.get("name", role), rho=float(data["rho"]),
                 tau0=float(data["tau0"]), k=float(data["k"]), n=float(data["n"]))


def load_case(path) -> CaseSpec:
    """Read a case file.  Unknown keys raise rather than being ignored."""
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: case file must be a JSON object")
    _check_keys("<root>", raw, _SECTIONS)

    fluids = raw.get("fluids")
    if not fluids:
        raise ValueError(f"{path}: a 'fluids' section is required")
    _check_keys("fluids", fluids, {"displaced", "displacing"})
    for role in ("displaced", "displacing"):
        if role not in fluids:
            raise ValueError(f"{path}: fluids.{role} is required")

    geometry = raw.get("geometry", {})
    _check_keys("geometry", geometry, _GEOMETRY_KEYS)
    flow = raw.get("flow", {})
    _check_keys("flow", flow, _FLOW_KEYS)
    interface = raw.get("interface", {})
    _check_keys("interface", interface, _INTERFACE_KEYS)
    rheology = raw.get("rheology", {})
    _check_keys("rheology", rheology, _RHEOLOGY_KEYS)

    gc = rheology.get("regularisation_shear_rate")
    return CaseSpec(
        displaced=_fluid_from("displaced", fluids["displaced"]),
        displacing=_fluid_from("displacing", fluids["displacing"]),
        name=raw.get("name", Path(path).stem),
        notes=raw.get("notes", ""),
        geometry={k: float(v) for k, v in geometry.items()},
        flow={k: float(v) for k, v in flow.items()},
        interface=InterfaceConfig(
            surface_tension=float(interface.get("surface_tension", 0.0))),
        regularisation_shear_rate=None if gc is None else float(gc),
        normalise_consistency=bool(rheology.get("normalise_consistency", False)),
    )


def save_case(spec: CaseSpec, path) -> Path:
    """Write a case file, so a modified setup can be kept alongside its results."""
    payload = {
        "name": spec.name,
        "notes": spec.notes,
        "fluids": {
            "displaced": asdict(spec.displaced),
            "displacing": asdict(spec.displacing),
        },
        "geometry": spec.geometry,
        "flow": spec.flow,
        "interface": {"surface_tension": spec.interface.surface_tension},
        "rheology": {
            "regularisation_shear_rate": spec.regularisation_shear_rate,
            "normalise_consistency": spec.normalise_consistency,
        },
    }
    path = Path(path)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


#: The Tao et al. (2025) setup, as shipped in ``cases/tao2025.json``.
DEFAULT_CASE = Path(__file__).resolve().parent.parent / "cases" / "tao2025.json"
