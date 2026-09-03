"""Case files - editable fluid properties, geometry and rheology."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from inpipe.caseio import DEFAULT_CASE, CaseSpec, load_case, save_case
from inpipe.fluid import Fluid

CASE_PATH = Path(__file__).resolve().parent.parent / "cases" / "kgep1.json"

MINIMAL = {
    "fluids": {
        "displaced": {"rho": 998.0, "tau0": 0.0, "k": 1e-3, "n": 1.0},
        "displacing": {"rho": 1200.0, "tau0": 1.4, "k": 0.6, "n": 0.4},
    }
}


def write(tmp_path, payload, name="case.json"):
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return path


def test_minimal_case_loads_with_defaults(tmp_path):
    spec = load_case(write(tmp_path, MINIMAL))
    assert spec.displaced.rho == 998.0
    assert spec.displacing.tau0 == 1.4
    assert spec.regularisation_shear_rate is None
    assert spec.normalise_consistency is False
    assert spec.interface.surface_tension == 0.0


def test_bundled_tao_case_matches_the_paper():
    spec = load_case(DEFAULT_CASE)
    assert spec.displacing.rho == 1200.0
    assert (spec.displacing.tau0, spec.displacing.k, spec.displacing.n) == (1.4, 0.6, 0.4)
    assert spec.displaced.rho == 998.0 and spec.displaced.is_newtonian
    assert spec.displaced.k == pytest.approx(1e-3)
    assert spec.interface.surface_tension == pytest.approx(0.07)
    assert spec.regularisation_shear_rate == pytest.approx(5.5)
    assert spec.geometry["casing_id"] == pytest.approx(0.16)
    assert spec.geometry["hole_diameter"] == pytest.approx(0.40)


# --- the point of the loader: typos must not pass silently ------------------


@pytest.mark.parametrize(
    "payload,match",
    [
        ({"fluids": {"displaced": {"rho": 1, "tau0": 0, "k": 1, "n": 1, "viscosity": 1},
                     "displacing": MINIMAL["fluids"]["displacing"]}}, "unknown key"),
        ({**MINIMAL, "geometry": {"lenght": 1.0}}, "unknown key"),
        ({**MINIMAL, "flow": {"velocity": 1.0}}, "unknown key"),
        ({**MINIMAL, "interface": {"sigma": 0.07}}, "unknown key"),
        ({**MINIMAL, "rheology": {"gamma_c": 5.5}}, "unknown key"),
        ({**MINIMAL, "rheolgy": {}}, "unknown key"),
    ],
)
def test_typos_are_rejected(tmp_path, payload, match):
    with pytest.raises(ValueError, match=match):
        load_case(write(tmp_path, payload))


def test_missing_sections_and_properties_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="'fluids' section is required"):
        load_case(write(tmp_path, {"geometry": {"length": 1.0}}))
    with pytest.raises(ValueError, match="fluids.displacing is required"):
        load_case(write(tmp_path, {"fluids": {"displaced": MINIMAL["fluids"]["displaced"]}}))
    with pytest.raises(ValueError, match=r"missing \['n'\]"):
        load_case(write(tmp_path, {"fluids": {
            "displaced": {"rho": 998.0, "tau0": 0.0, "k": 1e-3},
            "displacing": MINIMAL["fluids"]["displacing"]}}))


def test_non_object_file_is_rejected(tmp_path):
    path = tmp_path / "c.json"
    path.write_text("[1, 2, 3]")
    with pytest.raises(ValueError, match="must be a JSON object"):
        load_case(path)


def test_unphysical_properties_are_rejected_by_the_fluid_model(tmp_path):
    bad = {"fluids": {"displaced": {"rho": -1.0, "tau0": 0.0, "k": 1e-3, "n": 1.0},
                      "displacing": MINIMAL["fluids"]["displacing"]}}
    with pytest.raises(ValueError, match="density must be positive"):
        load_case(write(tmp_path, bad))


# --- round trip -------------------------------------------------------------


def test_save_then_load_round_trips(tmp_path):
    spec = load_case(DEFAULT_CASE)
    path = save_case(spec, tmp_path / "out.json")
    back = load_case(path)
    assert back.displacing == spec.displacing
    assert back.displaced == spec.displaced
    assert back.geometry == spec.geometry
    assert back.flow == spec.flow
    assert back.interface.surface_tension == spec.interface.surface_tension
    assert back.regularisation_shear_rate == spec.regularisation_shear_rate


def test_summary_reports_the_rheology_treatment():
    spec = load_case(DEFAULT_CASE)
    text = spec.summary()
    assert "Fluent-style regularisation" in text and "no plug" in text
    assert "reported, not modelled" in text

    exact = CaseSpec(displaced=Fluid.newtonian(998.0, 1e-3),
                     displacing=Fluid("c", 1200.0, 1.4, 0.6, 0.4))
    assert "rigid plug" in exact.summary()


def test_interface_dimensionless_groups():
    spec = load_case(DEFAULT_CASE)
    ca = spec.interface.capillary_number(0.854, 0.1067)
    assert ca == pytest.approx(0.854 * 0.1067 / 0.07, rel=1e-12)
    bo = spec.interface.bond_number(202.0, 0.1)
    assert bo == pytest.approx(202.0 * 9.80665 * 0.01 / 0.07, rel=1e-12)
    # Miscible: no interfacial tension, so the groups are unbounded.
    from inpipe.config import InterfaceConfig

    assert InterfaceConfig().capillary_number(1.0, 1.0) == float("inf")
    assert InterfaceConfig().bond_number(1.0, 1.0) == float("inf")


def test_the_field_case_reads_its_fluids_from_the_case_file(tmp_path):
    """Editing ``cases/kgep1.json`` must change what the well case simulates.

    The point of the case file is that the mud and slurry properties are not
    yet known, so they will be changed repeatedly.  If the case script kept its
    own constants, an edit would appear to work - the file parses, the summary
    prints the new numbers - and the simulation would quietly run the old ones.
    """
    from cases.circulation import CASE, build
    from inpipe.caliper import synthetic_caliper
    from inpipe.caseio import load_case, save_case
    from inpipe.config import inch_to_m

    edited = tmp_path / "edited.json"
    spec = load_case(CASE_PATH)
    save_case(
        replace(
            spec,
            displaced=Fluid("brine", 1010.0, 0.0, 1.1e-3, 1.0),
            displacing=Fluid("slurry", 1950.0, 9.0, 0.8, 0.55),
        ),
        edited,
    )

    caliper = synthetic_caliper(200.0, inch_to_m(8.5))
    solver, schedule, *_ = build(caliper, n_axial=12, top_depth=0.0,
                                 spec=load_case(edited))

    assert solver.fluids[0].name == "brine"
    assert solver.fluids[0].rho == 1010.0
    assert schedule.stages[0].fluid.tau0 == 9.0
    # And the default case is untouched by the edit.
    assert CASE.displaced.name == "mud"
