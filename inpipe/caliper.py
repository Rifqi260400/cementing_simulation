"""Caliper log handling - the wellbore is not a smooth cylinder.

A caliper log gives measured hole diameter against measured depth.  This module
reads one, resamples it onto the solver's axial grid, and synthesises a
plausible log when no measurement is available.

Accepted input
--------------
CSV with a header row naming a depth column and a diameter column.  Column
names are matched case-insensitively against common aliases, so all of these
work without configuration::

    DEPT,CALI          MD,HoleDia      depth_m,diameter_m
    100.0,8.62         328.1,8.62      30.48,0.2189

Units
-----
Getting units wrong is the worst failure mode a caliper reader has - it is
silent and it scales the whole well.  So they are resolved in this order, and
the choice is always recorded in :attr:`CaliperLog.units`:

1. An explicit ``depth_unit`` / ``diameter_unit`` argument.
2. A unit written in the column name (``depth_m``, ``MD_ft``, ``CALI_in``,
   ``diameter (m)``) or, for LAS, the unit field of the ``~C`` section.
3. Inference from the *diameter* only, which is unambiguous: a borehole is
   either ~0.2 m or ~8.5 in, two orders of magnitude apart, and no real hole
   is 8.5 m or 0.2 in wide.  The depth unit is then taken to match the
   diameter's unit system, because a log is internally consistent - a log
   reporting inches reports feet.

Depth is never inferred from its own magnitude.  A 650 ft well and a 650 m
well are indistinguishable that way, and guessing wrong scales every depth by
3.28.  If the rule above cannot settle it, the reader raises rather than
guesses.  A simple LAS file is also accepted - the ``~A`` data section is read
and the ``~C`` section supplies curve names and units.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import ft_to_m, inch_to_m

__all__ = ["CaliperLog", "read_caliper", "synthetic_caliper", "implausible_tail"]

_DEPTH_ALIASES = {"dept", "depth", "md", "measured_depth", "depth_m", "depth_ft", "tvd"}
_DIAMETER_ALIASES = {
    "cali", "cal", "caliper", "holedia", "hole_diameter", "hole_dia", "diameter",
    "diameter_m", "diameter_in", "d", "bit_size", "hd",
}

#: A median diameter above this is inches, below it is metres.  The gap
#: between the two readings of any real borehole is ~40x, so this is safe.
DIAMETER_IN_THRESHOLD = 1.0
#: Plausible borehole diameters once converted to metres.  Outside this range
#: the reader refuses rather than returning a nonsense well.
DIAMETER_RANGE_M = (0.02, 2.0)

_UNIT_SUFFIXES = {
    "m": "m", "metre": "m", "meter": "m", "metres": "m", "meters": "m",
    "ft": "ft", "feet": "ft", "foot": "ft", "f": "ft",
    "in": "in", "inch": "in", "inches": "in", "\"": "in",
}


def _unit_from_name(header: str):
    """Extract a unit written into a column name, or None."""
    import re

    token = header.strip().lower()
    for match in re.findall(r"[\(\[]([^)\]]+)[)\]]|_([a-z\"]+)$", token):
        for candidate in match:
            unit = _UNIT_SUFFIXES.get(candidate.strip())
            if unit is not None:
                return unit
    return None


@dataclass
class CaliperLog:
    """Hole diameter against measured depth, in SI.

    Attributes
    ----------
    depth : measured depth [m], strictly increasing.
    diameter : hole diameter [m] at each depth.
    name : label for plots and diagnostics.
    """

    depth: np.ndarray
    diameter: np.ndarray
    name: str = "caliper"
    #: How the units were resolved, for the record.
    units: str = "SI (constructed directly)"

    def __post_init__(self) -> None:
        self.depth = np.asarray(self.depth, dtype=float)
        self.diameter = np.asarray(self.diameter, dtype=float)
        if self.depth.ndim != 1 or self.depth.shape != self.diameter.shape:
            raise ValueError("depth and diameter must be 1D arrays of equal length")
        if self.depth.size < 2:
            raise ValueError("a caliper log needs at least two samples")
        order = np.argsort(self.depth)
        self.depth = self.depth[order]
        self.diameter = self.diameter[order]
        if np.any(np.diff(self.depth) <= 0.0):
            raise ValueError("caliper depths must be strictly increasing")
        if np.any(self.diameter <= 0.0):
            raise ValueError("caliper diameters must be positive")

    def diameter_at(self, z) -> np.ndarray:
        """Hole diameter [m] at arbitrary depths, linearly interpolated.

        Depths outside the log are held at the nearest logged value rather than
        extrapolated - a caliper says nothing about what lies beyond it.
        """
        return np.interp(np.asarray(z, dtype=float), self.depth, self.diameter)

    def resample(self, z) -> np.ndarray:
        return self.diameter_at(z)

    @property
    def gauge(self) -> float:
        """Modal ("in-gauge") diameter [m], taken as the median."""
        return float(np.median(self.diameter))

    def washout_fraction(self, tolerance: float = 1.02) -> float:
        """Fraction of the logged interval wider than ``tolerance`` x gauge."""
        return float(np.mean(self.diameter > tolerance * self.gauge))

    def excess_volume(self, casing_od: float) -> float:
        """Annular volume above what an in-gauge hole would hold [m^3]."""
        r_c = 0.5 * casing_od
        area = math.pi * ((0.5 * self.diameter) ** 2 - r_c**2)
        gauge_area = math.pi * ((0.5 * self.gauge) ** 2 - r_c**2)
        return float(np.trapezoid(area - gauge_area, self.depth))

    def summary(self, casing_od: float | None = None) -> str:
        lines = [
            f"caliper {self.name!r}: {self.depth.size} samples, "
            f"{self.depth[0]:.1f} - {self.depth[-1]:.1f} m MD",
            f"  units: {self.units}",
            f"  diameter: min {self.diameter.min() * 1e3:.1f} mm, "
            f"gauge {self.gauge * 1e3:.1f} mm, max {self.diameter.max() * 1e3:.1f} mm",
            f"  washed out (> 2 % over gauge): "
            f"{100 * self.washout_fraction():.1f} % of the interval",
        ]
        if casing_od is not None:
            lines.append(f"  excess annular volume: {self.excess_volume(casing_od):.4f} m^3")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


def _strip_unit_suffix(token: str) -> str:
    """``"depth_ft"`` -> ``"depth"``, ``"cali (in)"`` -> ``"cali"``."""
    import re

    token = re.sub(r"\s*[\(\[][^)\]]*[)\]]\s*$", "", token).strip()
    base, sep, suffix = token.rpartition("_")
    if sep and suffix in _UNIT_SUFFIXES:
        return base
    return token


def _pick_column(headers, aliases, kind, exclude=()):
    """Find the column for ``kind``, never returning one in ``exclude``.

    Matching is exact first (on the raw name, then with any unit suffix
    stripped), and only then by substring - and substring matching ignores
    aliases shorter than three characters, because a name like ``"d"`` occurs
    inside almost every header and would happily claim the depth column as the
    diameter.
    """
    lowered = [h.strip().lower() for h in headers]
    stripped = [_strip_unit_suffix(h) for h in lowered]
    long_aliases = {a for a in aliases if len(a) >= 3}

    for candidates in (lowered, stripped):
        for i, h in enumerate(candidates):
            if i not in exclude and h in aliases:
                return i
    for i, h in enumerate(stripped):
        if i not in exclude and any(a in h for a in long_aliases):
            return i
    raise ValueError(
        f"could not find a {kind} column in {headers!r}; "
        f"rename it to one of {sorted(aliases)[:5]} or pass the column index"
    )


def _convert_units(depth, diameter, depth_unit, diameter_unit, source=""):
    """Resolve and apply units.  See the module docstring for the rules."""
    if diameter_unit is None:
        diameter_unit = (
            "in" if float(np.median(diameter)) > DIAMETER_IN_THRESHOLD else "m"
        )
        diameter_source = "inferred from magnitude"
    else:
        diameter_source = "given"

    if depth_unit is None:
        # A log is internally consistent: inches go with feet, metres with
        # metres.  Depth magnitude is never used - see the module docstring.
        depth_unit = "ft" if diameter_unit == "in" else "m"
        depth_source = f"assumed to match the diameter unit ({diameter_unit})"
    else:
        depth_source = "given"

    if depth_unit == "ft":
        depth = ft_to_m(depth)
    elif depth_unit != "m":
        raise ValueError(f"unknown depth unit {depth_unit!r}; expected 'm' or 'ft'")
    if diameter_unit == "in":
        diameter = inch_to_m(diameter)
    elif diameter_unit != "m":
        raise ValueError(f"unknown diameter unit {diameter_unit!r}; expected 'm' or 'in'")

    lo, hi = DIAMETER_RANGE_M
    med = float(np.median(diameter))
    if not lo <= med <= hi:
        raise ValueError(
            f"{source}median hole diameter reads {med:.4g} m after unit resolution "
            f"(diameter unit {diameter_unit!r}, {diameter_source}), which is outside "
            f"the plausible range {lo}-{hi} m. Pass diameter_unit= explicitly."
        )
    units = (
        f"depth {depth_unit} ({depth_source}), diameter {diameter_unit} "
        f"({diameter_source})"
    )
    return depth, diameter, units


def read_caliper(
    path,
    depth_column=None,
    diameter_column=None,
    depth_unit=None,
    diameter_unit=None,
    null_value=-999.25,
    depth_min=None,
    depth_max=None,
    min_diameter=None,
) -> CaliperLog:
    """Read a caliper log from CSV or LAS.

    Columns are found by name unless ``depth_column`` / ``diameter_column`` are
    given (name or zero-based index).  Units are inferred unless stated; see the
    module docstring.  Rows whose diameter equals ``null_value`` or is
    non-finite are dropped; a LAS file's own ``NULL`` declaration overrides the
    argument.

    ``depth_min`` / ``depth_max`` trim the log to a depth window [m], and
    ``min_diameter`` drops samples narrower than a threshold [m].  Both are for
    cutting the junk a real log carries - a collapsed caliper arm at total
    depth, say.  Nothing is trimmed automatically: use
    :func:`implausible_tail` to find where a log goes bad, then say so
    explicitly, so the cut is a recorded decision rather than a silent one.
    """
    path = Path(path)
    # Real logs carry stray bytes in curve descriptions (degree signs, and so
    # on).  Those never matter to the numbers, so do not let them stop the read.
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() == ".las" or text.lstrip().startswith("~"):
        headers, rows, declared_null = _parse_las(text)
        if declared_null is not None:
            null_value = declared_null
    else:
        headers, rows = _parse_csv(text)

    di = _resolve_column(headers, depth_column, _DEPTH_ALIASES, "depth")
    ci = _resolve_column(headers, diameter_column, _DIAMETER_ALIASES, "diameter",
                         exclude=(di,))
    if ci == di:
        raise ValueError(
            f"{path.name}: depth and diameter resolved to the same column "
            f"{headers[di]!r}; pass depth_column= and diameter_column= explicitly"
        )

    # A unit written into the column name beats inference.
    if depth_unit is None:
        depth_unit = _unit_from_name(headers[di])
        if depth_unit == "in":  # a depth is never in inches
            depth_unit = None
    if diameter_unit is None:
        diameter_unit = _unit_from_name(headers[ci])
        if diameter_unit == "ft":  # a hole diameter is never in feet
            diameter_unit = None

    depth, diameter = [], []
    for row in rows:
        if len(row) <= max(di, ci):
            continue
        try:
            d, c = float(row[di]), float(row[ci])
        except ValueError:
            continue
        if not (math.isfinite(d) and math.isfinite(c)):
            continue
        if null_value is not None and (c == null_value or d == null_value):
            continue
        depth.append(d)
        diameter.append(c)

    if len(depth) < 2:
        raise ValueError(f"{path}: fewer than two usable caliper samples")

    depth = np.array(depth)
    diameter = np.array(diameter)
    if depth_min is not None:
        keep = depth >= depth_min
        depth, diameter = depth[keep], diameter[keep]
    if depth_max is not None:
        keep = depth <= depth_max
        depth, diameter = depth[keep], diameter[keep]
    if depth.size < 2:
        raise ValueError(f"{path}: fewer than two samples left after trimming")
    depth, diameter, units = _convert_units(
        depth, diameter, depth_unit, diameter_unit, source=f"{path.name}: "
    )
    # Collapse duplicate depths, keeping the mean diameter.
    uniq, inverse = np.unique(depth, return_inverse=True)
    if uniq.size != depth.size:
        summed = np.zeros(uniq.size)
        counts = np.zeros(uniq.size)
        np.add.at(summed, inverse, diameter)
        np.add.at(counts, inverse, 1.0)
        depth, diameter = uniq, summed / counts
    if min_diameter is not None:
        keep = diameter >= min_diameter
        if keep.sum() < 2:
            raise ValueError(
                f"{path}: fewer than two samples at or above min_diameter="
                f"{min_diameter:.4g} m"
            )
        depth, diameter = depth[keep], diameter[keep]
    return CaliperLog(depth, diameter, name=path.name, units=units)


def _resolve_column(headers, given, aliases, kind, exclude=()):
    if given is None:
        return _pick_column(headers, aliases, kind, exclude=exclude)
    if isinstance(given, int):
        return given
    lowered = [h.strip().lower() for h in headers]
    try:
        return lowered.index(str(given).strip().lower())
    except ValueError:
        raise ValueError(f"{kind} column {given!r} not in {headers!r}") from None


def _parse_csv(text):
    rows = list(csv.reader(text.splitlines()))
    rows = [r for r in rows if r and any(c.strip() for c in r)]
    if not rows:
        raise ValueError("empty CSV")
    return rows[0], rows[1:]


def _parse_las(text):
    """LAS 2.0 reader covering both WRAP modes.

    ``~V`` gives the wrap flag, ``~W`` the NULL value, ``~C`` the curve
    mnemonics and units, ``~A`` the data.  In wrapped files one depth step
    spans several lines, so the data section is tokenised as a flat stream and
    reshaped by the curve count - which also handles the unwrapped case.

    Returns ``(headers, rows, null_value)``; ``null_value`` is ``None`` when the
    file does not declare one.
    """
    headers, tokens, section = [], [], None
    null_value = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("~"):
            section = line[1].upper()
            continue
        if section == "W" and line.upper().startswith("NULL"):
            for token in line.split(":")[0].split():
                try:
                    null_value = float(token)
                    break
                except ValueError:
                    continue
        elif section == "C":
            mnemonic, _, rest = line.partition(".")
            parts = rest.split()
            unit = parts[0].split(":")[0].strip() if parts else ""
            headers.append(f"{mnemonic.strip()}_{unit}" if unit else mnemonic.strip())
        elif section == "A":
            tokens.extend(line.split())

    if not headers:
        raise ValueError("LAS file has no ~C curve section")
    n = len(headers)
    complete = (len(tokens) // n) * n
    if complete == 0:
        raise ValueError("LAS file has no complete data records")
    rows = [tokens[i:i + n] for i in range(0, complete, n)]
    return headers, rows, null_value


def implausible_tail(log: "CaliperLog", fraction: float = 0.6):
    """Find a contiguous run of implausibly narrow hole at the bottom of a log.

    A caliper often collapses at total depth - the arms close on fill, or the
    tool bottoms out - leaving a block of readings far below gauge that is
    instrument behaviour, not geometry.

    Returns ``(start_depth, end_depth, n_samples)`` for the trailing run below
    ``fraction`` x gauge, or ``None`` if the log ends in gauge.  The caller
    decides whether to cut; this only reports.
    """
    threshold = fraction * log.gauge
    below = log.diameter < threshold
    if not below[-1]:
        return None
    # Walk back while the readings stay below threshold.
    i = len(below) - 1
    while i > 0 and below[i - 1]:
        i -= 1
    return float(log.depth[i]), float(log.depth[-1]), int(len(below) - i)


# ---------------------------------------------------------------------------
# Synthetic log, for when no measurement is available
# ---------------------------------------------------------------------------


def synthetic_caliper(
    length: float,
    gauge_diameter: float,
    washouts=((0.28, 0.10, 1.55), (0.55, 0.06, 1.30), (0.78, 0.13, 1.72)),
    tight_zones=((0.42, 0.04, 0.94),),
    n_samples: int = 2001,
    roughness: float = 0.015,
    seed: int = 0,
    name: str = "synthetic",
) -> CaliperLog:
    """Build a plausible caliper log for a well of the given length.

    Each entry in ``washouts`` and ``tight_zones`` is
    ``(centre_fraction, width_fraction, diameter_multiplier)``, with fractions
    of the total length.  Zones are laid on with a raised-cosine profile so the
    hole diameter is continuous, plus band-limited roughness of relative
    amplitude ``roughness``.

    This is a stand-in, clearly labelled as such - swap in
    :func:`read_caliper` as soon as a real log is available.
    """
    z = np.linspace(0.0, length, n_samples)
    d = np.full_like(z, gauge_diameter)

    for centre, width, mult in list(washouts) + list(tight_zones):
        c, w = centre * length, max(width * length, 1e-9)
        window = np.clip((z - c) / (0.5 * w), -1.0, 1.0)
        bump = 0.5 * (1.0 + np.cos(math.pi * window))  # 1 at centre, 0 at edges
        d = d + gauge_diameter * (mult - 1.0) * bump

    if roughness > 0.0:
        rng = np.random.default_rng(seed)
        noise = rng.standard_normal(n_samples)
        # Smooth to a realistic vertical correlation length (~1 % of the well).
        window = max(int(0.01 * n_samples), 3)
        kernel = np.ones(window) / window
        noise = np.convolve(noise, kernel, mode="same")
        noise /= max(np.std(noise), 1e-30)
        d = d * (1.0 + roughness * noise)

    return CaliperLog(z, np.maximum(d, 0.5 * gauge_diameter), name=name)
