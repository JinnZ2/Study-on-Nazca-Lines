#!/usr/bin/env python3
"""
example_shadow_study.py — Build your own shadow analysis

This script shows how to use the shadow library to study shadow behaviour
at any location on Earth.  Edit the CONFIGURATION section below, run the
script, and it will produce a CSV table and (optionally) PNG plots.

Usage:
    python examples/example_shadow_study.py

Requires:
    pip install -r requirements.txt   (only matplotlib, and only for plots)
"""

import sys
from pathlib import Path
from datetime import date

# Allow running from the examples/ directory or repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shadow import (
    solar_position,
    standard_refraction_deg,
    mirage_shift_deg,
    curvature_tilt_deg,
    effective_altitude_deg,
    shadow_length_m,
    ground_accuracy_m,
    simulate_day,
    write_csv,
    ShadowInputs,
)

# ============================================================
# CONFIGURATION — change these to study your own location/date
# ============================================================

# Location (latitude +N, longitude +E)
LATITUDE = -14.735      # Nazca Pampa, Peru
LONGITUDE = -75.13

# Date to simulate (UTC)
SIM_DATE = date(2025, 9, 21)  # September equinox

# Time window (UTC hours) and sampling resolution
HOUR_START = 12   # 12:00 UTC
HOUR_END = 22     # 22:00 UTC
STEP_MINUTES = 10

# Object height (metres) — the thing casting the shadow
OBJECT_HEIGHT_M = 1.5  # e.g. a person or a stake

# Atmosphere
PRESSURE_HPA = 1010.0  # sea-level default
TEMPERATURE_C = 25.0    # warm desert day

# Mirage parameters
#   temp_gradient: near-surface dT/dz in K/m (negative = hot ground)
#   layer_thickness: depth of the thermal layer in metres
#   coeff: empirical sensitivity in deg/K (tune with field data)
TEMP_GRADIENT = -0.15   # stronger desert gradient
MIRAGE_LAYER_M = 2.0
MIRAGE_COEFF = 0.02

# Curvature baseline — distance over which ground curves away
BASELINE_M = 1000.0     # 1 km sightline
EARTH_RADIUS_M = 6_371_000.0

# Output files
OUT_CSV = "my_shadow_study.csv"
OUT_PNG = "my_shadow_study.png"  # set to None to skip plots

# Textile-to-ground projection (optional Nazca geoglyph analysis)
#   thread_spacing_mm: distance between threads in the textile model
#   scale_factor: enlargement from textile to ground
THREAD_SPACING_MM = 0.8
GROUND_SCALE_FACTOR = 200.0

# ============================================================
# RUN — you shouldn't need to change anything below
# ============================================================

def main():
    # --- 1. Build the physics inputs ---
    inputs = ShadowInputs(
        height_m=OBJECT_HEIGHT_M,
        pressure_hPa=PRESSURE_HPA,
        temp_C=TEMPERATURE_C,
        temp_gradient_K_per_m=TEMP_GRADIENT,
        mirage_layer_m=MIRAGE_LAYER_M,
        mirage_coeff_deg_per_K=MIRAGE_COEFF,
        baseline_m=BASELINE_M,
        radius_m=EARTH_RADIUS_M,
    )

    # --- 2. Run the day-long simulation ---
    samples = simulate_day(
        lat_deg=LATITUDE,
        lon_deg=LONGITUDE,
        day=SIM_DATE,
        hours_from=HOUR_START,
        hours_to=HOUR_END,
        step_min=STEP_MINUTES,
        inputs=inputs,
    )

    # --- 3. Write results to CSV ---
    write_csv(samples, OUT_CSV)
    print(f"Wrote {len(samples)} samples to {OUT_CSV}")

    # --- 4. Print a summary table ---
    print()
    print(f"{'Time (UTC)':<22} {'Sun Alt':>8} {'Refr':>7} {'Mirage':>7} "
          f"{'Curv':>7} {'a_eff':>7} {'Shadow':>9}")
    print("-" * 75)
    for s in samples:
        length_str = f"{s.shadow_length_m:.2f} m" if s.shadow_length_m is not None else "   --"
        print(f"{s.t_utc.strftime('%Y-%m-%d %H:%M'):<22} "
              f"{s.sun_alt_deg:>7.2f}° {s.refraction_deg:>6.4f}° {s.mirage_deg:>+6.4f}° "
              f"{s.curvature_deg:>6.4f}° {s.alpha_eff_deg:>7.2f}° {length_str:>9}")

    # --- 5. Show individual correction functions (for learning) ---
    print()
    print("=== Individual corrections at a single moment ===")
    from datetime import datetime, timezone
    moment = datetime(SIM_DATE.year, SIM_DATE.month, SIM_DATE.day, 17, 0, tzinfo=timezone.utc)
    az, alt = solar_position(LATITUDE, LONGITUDE, moment)
    refr = standard_refraction_deg(alt, PRESSURE_HPA, TEMPERATURE_C)
    mir = mirage_shift_deg(TEMP_GRADIENT, MIRAGE_LAYER_M, MIRAGE_COEFF)
    curv = curvature_tilt_deg(BASELINE_M, EARTH_RADIUS_M)
    a_eff = effective_altitude_deg(alt, refr, mir, curv)
    shadow = shadow_length_m(OBJECT_HEIGHT_M, a_eff)

    print(f"  Time:          {moment.isoformat()}")
    print(f"  Sun azimuth:   {az:.2f}°  (from north)")
    print(f"  Sun altitude:  {alt:.2f}°")
    print(f"  Refraction:   +{refr:.4f}°  (lifts apparent sun)")
    print(f"  Mirage:       {mir:+.4f}°  (thermal layer shift)")
    print(f"  Curvature:    -{curv:.6f}°  (ground falls away over {BASELINE_M:.0f} m)")
    print(f"  Effective alt: {a_eff:.2f}°")
    if shadow is not None:
        print(f"  Shadow length: {shadow:.3f} m")
    else:
        print(f"  Shadow length: infinite (sun at/below horizon)")

    # --- 6. Textile-to-ground projection ---
    accuracy = ground_accuracy_m(THREAD_SPACING_MM, GROUND_SCALE_FACTOR)
    print()
    print("=== Textile-to-ground projection ===")
    print(f"  Thread spacing:    {THREAD_SPACING_MM} mm")
    print(f"  Scale factor:      {GROUND_SCALE_FACTOR}x")
    print(f"  Ground accuracy:   {accuracy:.3f} m  ({accuracy*100:.1f} cm)")

    # --- 7. Optional plots ---
    if OUT_PNG is not None:
        try:
            import matplotlib.pyplot as plt

            times = [s.t_utc for s in samples]
            alts = [s.sun_alt_deg for s in samples]
            effs = [s.alpha_eff_deg for s in samples]
            lengths = [s.shadow_length_m if s.shadow_length_m is not None
                       else float("nan") for s in samples]

            fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

            axes[0].plot(times, alts, label="True altitude")
            axes[0].plot(times, effs, label="Effective altitude", linestyle="--")
            axes[0].set_ylabel("Degrees")
            axes[0].set_title("Solar Altitude")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            axes[1].plot(times, [s.refraction_deg for s in samples], label="Refraction")
            axes[1].plot(times, [s.mirage_deg for s in samples], label="Mirage")
            axes[1].axhline(y=samples[0].curvature_deg, color="gray",
                            linestyle=":", label="Curvature tilt")
            axes[1].set_ylabel("Degrees")
            axes[1].set_title("Correction Terms")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            axes[2].plot(times, lengths)
            axes[2].set_ylabel("Metres")
            axes[2].set_xlabel("UTC Time")
            axes[2].set_title("Shadow Length")
            axes[2].grid(True, alpha=0.3)

            fig.suptitle(
                f"Shadow Study: {LATITUDE}°N {LONGITUDE}°E — {SIM_DATE}",
                fontsize=12, fontweight="bold",
            )
            plt.tight_layout()
            plt.savefig(OUT_PNG, dpi=160)
            plt.close()
            print(f"\nPlot saved to {OUT_PNG}")

        except ImportError:
            print("\nmatplotlib not installed — skipping plots.")
            print("Install with:  pip install matplotlib")


if __name__ == "__main__":
    main()
