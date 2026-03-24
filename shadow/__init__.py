# SPDX-License-Identifier: MIT
"""shadow — Shadow projection physics (geometry + mirage + curvature)."""

from shadow.shadow import (
    solar_position,
    standard_refraction_deg,
    mirage_shift_deg,
    curvature_tilt_deg,
    effective_altitude_deg,
    shadow_length_m,
    simulate_day,
    write_csv,
    ground_accuracy_m,
    ShadowInputs,
    ShadowSample,
)

__all__ = [
    "solar_position",
    "standard_refraction_deg",
    "mirage_shift_deg",
    "curvature_tilt_deg",
    "effective_altitude_deg",
    "shadow_length_m",
    "simulate_day",
    "write_csv",
    "ground_accuracy_m",
    "ShadowInputs",
    "ShadowSample",
]
