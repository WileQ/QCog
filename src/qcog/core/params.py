"""Model parameters as data.

Every Q-Cog model declares its parameters once, as :class:`ParamSpec` objects.
That single declaration then drives

* the interactive explorer (widgets and sliders are generated from it),
* the command line (``--theta 0.9``),
* the fitting layer (bounds and starting points come from it),
* the generated documentation (the parameter tables), and
* reproducibility (a run is fully described by its parameter dictionary).

Declaring parameters once and deriving everything else from them is the main
reason the library scales to twenty models without twenty ad-hoc user interfaces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

__all__ = ["ParamSpec", "coerce_params", "fit_bounds", "parameter_table", "KINDS"]

KINDS = ("float", "int", "bool", "choice", "text")


@dataclass(frozen=True)
class ParamSpec:
    """A single model parameter.

    ``low``/``high`` are the *physical* bounds: values outside them are clamped,
    and the fitting layer never leaves them.
    """

    name: str
    kind: str = "float"
    default: Any = 0.0
    label: str = ""
    description: str = ""
    low: float | None = None
    high: float | None = None
    step: float | None = None
    choices: tuple[Any, ...] | None = None
    unit: str = ""
    fittable: bool = False

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown parameter kind {self.kind!r}; expected one of {KINDS}")
        if self.kind == "choice" and not self.choices:
            raise ValueError(f"parameter {self.name!r} of kind 'choice' needs choices")

    @property
    def display(self) -> str:
        return self.label or self.name.replace("_", " ")

    def coerce(self, value: Any) -> Any:
        """Convert user input into the declared type, clamped to the declared bounds."""
        if value is None:
            return self.default
        if self.kind == "float":
            number = float(value)
            if self.low is not None:
                number = max(number, float(self.low))
            if self.high is not None:
                number = min(number, float(self.high))
            return float(number)
        if self.kind == "int":
            number = int(round(float(value)))
            if self.low is not None:
                number = max(number, int(self.low))
            if self.high is not None:
                number = min(number, int(self.high))
            return int(number)
        if self.kind == "bool":
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "on", "y", "t"}
            return bool(value)
        if self.kind == "choice":
            if value not in self.choices:
                raise ValueError(
                    f"parameter {self.name!r} must be one of {list(self.choices)}, got {value!r}"
                )
            return value
        return str(value)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "default": self.default,
            "label": self.display,
            "description": self.description,
            "low": self.low,
            "high": self.high,
            "step": self.step,
            "choices": list(self.choices) if self.choices else None,
            "unit": self.unit,
            "fittable": self.fittable,
        }


def coerce_params(
    specs: Sequence[ParamSpec],
    values: Mapping[str, Any] | None = None,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Merge user-supplied values with the declared defaults.

    With ``strict=True`` (the default) an unknown parameter name raises, which
    catches typos in scripts and in the command line early.
    """
    values = dict(values or {})
    known = {spec.name for spec in specs}
    if strict:
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"unknown parameter(s) {unknown}; expected any of {sorted(known)}")
    resolved: dict[str, Any] = {}
    for spec in specs:
        resolved[spec.name] = spec.coerce(values.get(spec.name, spec.default))
    return resolved


def fit_bounds(specs: Iterable[ParamSpec]) -> list[tuple[float, float]]:
    """Bounds and order for the fittable numeric parameters of a model."""
    bounds = []
    for spec in specs:
        if not spec.fittable:
            continue
        low = -10.0 if spec.low is None else float(spec.low)
        high = 10.0 if spec.high is None else float(spec.high)
        bounds.append((low, high))
    return bounds


def parameter_table(specs: Sequence[ParamSpec]) -> list[dict]:
    """Rows describing every parameter, used in the docs and the UI."""
    return [spec.to_dict() for spec in specs]