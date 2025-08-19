# SPDX-License-Identifier: MIT
"""
shadow.py — Shadow Projection Physics (geometry + mirage + curvature)

Models shadow length/displacement for a vertical stake on (approximately) level terrain,
including: solar geometry, standard atmospheric refraction, an adjustable mirage layer,
and a simple curvature tilt term over a chosen baseline.

This is a planning/intuition tool (degree-level accuracy), not an astro-geodesy package.
"""

from __future__ import annotations
import math, csv
from dataclasses import dataclass
from datetime import datetime, date, time, timedelta, timezone
from typing import List, Tuple, Optional

# -------------------------------
# Solar position (approx. NOAA)
# -------------------------------

def _julian_day(dt_utc: datetime) -> float:
    y, m = dt_utc.year, dt_utc.month
    D = dt_utc.day + (dt_utc.hour + dt_utc.minute/60 + dt_utc.second/3600)/24.0
    if m <= 2:
        y -= 1; m += 12
    A = y // 100
    B = 2 - A + A // 4
    jd = int(365.25*(y + 4716)) + int(30.6001*(m + 1)) + D + B - 1524.5
    return jd

def solar_position(lat_deg: float, lon_deg: float, dt_utc: datetime) -> Tuple[float,float]:
    """Return (azimuth_deg_from_north, altitude_deg)."""
    jd = _julian_day(dt_utc)
    T = (jd - 2451545.0) / 36525.0

    L0 = (280.46646 + 36000.76983*T + 0.0003032*T*T) % 360.0
    M  = 357.52911 + 35999.05029*T - 0.0001537*T*T
    e  = 0.016708634 - 0.000042037*T - 0.0000001267*T*T

    Mrad = math.radians(M)
    C = (1.914602 - 0.004817*T - 0.000014*T*T)*math.sin(Mrad) \
        + (0.019993 - 0.000101*T)*math.sin(2*Mrad) \
        + 0.000289*math.sin(3*Mrad)

    true_long = L0 + C
    omega = 125.04 - 1934.136*T
    lam = true_long - 0.00569 - 0.00478*math.sin(math.radians(omega))

    # Obliquity
    eps0 = 23 + (26 + ((21.448 - (46.8150*T + 0.00059*T*T - 0.001813*T*T*T))/60.0))/60.0
    eps = eps0 + 0.00256*math.cos(math.radians(omega))

    # Declination
    decl = math.degrees(math.asin(math.sin(math.radians(eps))*math.sin(math.radians(lam))))

    # Equation of time (minutes)
    y = math.tan(math.radians(eps/2))**2
    E = 4*math.degrees(
        y*math.sin(2*math.radians(L0)) - 2*e*math.sin(Mrad) + 4*e*y*math.sin(Mrad)*math.cos(2*math.radians(L0))
        - 0.5*y*y*math.sin(4*math.radians(L0)) - 1.25*e*e*math.sin(2*Mrad)
    )

    minutes_utc = dt_utc.hour*60 + dt_utc.minute + dt_utc.second/60.0
    tst_minutes = (minutes_utc + E + 4.0*lon_deg) % 1440.0
    H = tst_minutes/4.0 - 180.0  # deg

    lat = math.radians(lat_deg)
    decl_r = math.radians(decl)
    H_r = math.radians(H)

    sin_alt = math.sin(lat)*math.sin(decl_r) + math.cos(lat)*math.cos(decl_r)*math.cos(H_r)
    alt = math.degrees(math.asin(max(-1.0, min(1.0, sin_alt))))

    az_r = math.atan2(
        math.sin(H_r),
        math.cos(H_r)*math.sin(lat) - math.tan(decl_r)*math.cos(lat)
    )
    # Convert to North-based azimuth
    az = (math.degrees(az_r) + 180.0) % 360.0
    return az, alt

# -------------------------------
# Refraction + Mirage models
# -------------------------------

def standard_refraction_deg(alt_deg: float, pressure_hPa: float=1010.0, temp_C: float=10.0) -> float:
    """
    Approximate astronomical refraction in degrees for apparent altitude alt_deg (sea-level default).
    Valid for altitudes > ~-1 deg. Bennett 1982-like formula.
    """
    if alt_deg < -1.0:
        return 0.0
    alt = math.radians(alt_deg + 7.31/(alt_deg + 4.4))  # avoid 90° near horizon
    R = (0.0002967 * pressure_hPa / (273.15 + temp_C)) / math.tan(alt)  # in radians
    return math.degrees(R)

def mirage_shift_deg(temp_gradient_K_per_m: float=-0.1, layer_thickness_m: float=2.0, coeff_deg_per_K: float=0.02) -> float:
    """
    Crude mirage angular shift model:
    Δθ_m (deg) ≈ coeff_deg_per_K * ΔT, where ΔT = (dT/dz) * layer_thickness.
    Sign convention: negative gradient (hot surface) -> inferior mirage -> apparent lowering (positive Δθ_m).
    TUNE coeff as needed with field data.
    """
    dT = temp_gradient_K_per_m * layer_thickness_m
    return coeff_deg_per_K * dT

def curvature_tilt_deg(baseline_m: float=1000.0, radius_m: float=6371000.0) -> float:
    """
    Effective tilt in degrees due to ground curvature across a baseline.
    For small angles, tilt ≈ baseline/(2R) radians (sagitta slope midspan).
    """
    tilt_rad = baseline_m/(2.0*radius_m)
    return math.degrees(tilt_rad)

# -------------------------------
# Shadow computation
# -------------------------------

@dataclass
class ShadowInputs:
    height_m: float
    pressure_hPa: float = 1010.0
    temp_C: float = 20.0
    temp_gradient_K_per_m: float = -0.1
    mirage_layer_m: float = 2.0
    mirage_coeff_deg_per_K: float = 0.02
    baseline_m: float = 1000.0
    radius_m: float = 6371000.0

@dataclass
class ShadowSample:
    t_utc: datetime
    sun_az_deg: float
    sun_alt_deg: float
    refraction_deg: float
    mirage_deg: float
    curvature_deg: float
    alpha_eff_deg: float
    shadow_length_m: Optional[float]

def effective_altitude_deg(alt_deg: float, refraction_deg: float, mirage_deg: float, curvature_deg: float) -> float:
    """
    Combine effects: α_eff = alt + refraction + mirage - curvature_tilt
    (Curvature reduces effective elevation as ground "falls away".)
    """
    return alt_deg + refraction_deg + mirage_deg - curvature_deg

def shadow_length_m(height_m: float, alpha_deg: float) -> Optional[float]:
    if alpha_deg <= 0.0:
        return None  # shadow goes to "infinite" / no intersection on ground plane in the model
    return height_m / math.tan(math.radians(alpha_deg))

def simulate_day(lat_deg: float, lon_deg: float, day: date, hours_from: int, hours_to: int,
                 step_min: int, inputs: ShadowInputs, tz_offset_hours: float=0.0) -> List[ShadowSample]:
    samples: List[ShadowSample] = []
    tz = timezone.utc
    t = datetime(day.year, day.month, day.day, hours_from, 0, tzinfo=tz)
    end = datetime(day.year, day.month, day.day, hours_to, 0, tzinfo=tz)
    curv = curvature_tilt_deg(inputs.baseline_m, inputs.radius_m)
    while t <= end:
        az, alt = solar_position(lat_deg, lon_deg, t)
        R = standard_refraction_deg(alt, inputs.pressure_hPa, inputs.temp_C)
        M = mirage_shift_deg(inputs.temp_gradient_K_per_m, inputs.mirage_layer_m, inputs.mirage_coeff_deg_per_K)
        alpha_eff = effective_altitude_deg(alt, R, M, curv)
        L = shadow_length_m(inputs.height_m, alpha_eff)
        samples.append(ShadowSample(t, az, alt, R, M, curv, alpha_eff, L))
        t += timedelta(minutes=step_min)
    return samples

def write_csv(samples: List[ShadowSample], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time_utc","sun_az_deg","sun_alt_deg","refraction_deg","mirage_deg",
                    "curvature_deg","alpha_eff_deg","shadow_length_m"])
        for s in samples:
            w.writerow([s.t_utc.isoformat(), f"{s.sun_az_deg:.4f}", f"{s.sun_alt_deg:.4f}",
                        f"{s.refraction_deg:.4f}", f"{s.mirage_deg:.4f}", f"{s.curvature_deg:.6f}",
                        f"{s.alpha_eff_deg:.4f}", "" if s.shadow_length_m is None else f"{s.shadow_length_m:.4f}"])

# -------------------------------
# Textile resolution utility
# -------------------------------

def ground_accuracy_m(thread_spacing_mm: float, scale_factor: float) -> float:
    """Projection accuracy ≈ d × scale (converted mm→m)."""
    return (thread_spacing_mm/1000.0) * scale_factor

# -------------------------------
# Simple CLI
# -------------------------------

def _plot(samples: List[ShadowSample], out_png: str) -> None:
    import matplotlib.pyplot as plt
    times = [s.t_utc for s in samples]
    elev = [s.sun_alt_deg for s in samples]
    alpha = [s.alpha_eff_deg for s in samples]
    L = [s.shadow_length_m if s.shadow_length_m is not None else float("nan") for s in samples]

    plt.figure(figsize=(8,4))
    plt.plot(times, elev)
    plt.title("Sun Elevation (deg)")
    plt.xlabel("UTC time")
    plt.ylabel("Elevation (deg)")
    plt.tight_layout()
    plt.savefig(out_png.replace(".png","_sun.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(8,4))
    plt.plot(times, alpha)
    plt.title("Effective Altitude α_eff (deg)")
    plt.xlabel("UTC time")
    plt.ylabel("α_eff (deg)")
    plt.tight_layout()
    plt.savefig(out_png.replace(".png","_alpha.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(8,4))
    plt.plot(times, L)
    plt.title("Shadow Length (m)")
    plt.xlabel("UTC time")
    plt.ylabel("Length (m)")
    plt.tight_layout()
    plt.savefig(out_png.replace(".png","_shadow.png"), dpi=160)
    plt.close()

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Shadow projection simulator (geometry + mirage + curvature)")
    ap.add_argument("--lat", type=float, required=True, help="Latitude (+N)")
    ap.add_argument("--lon", type=float, required=True, help="Longitude (+E)")
    ap.add_argument("--date", type=str, required=True, help="YYYY-MM-DD (UTC)")
    ap.add_argument("--from-hour", type=int, default=10, help="Start hour UTC")
    ap.add_argument("--to-hour", type=int, default=20, help="End hour UTC")
    ap.add_argument("--step-min", type=int, default=2, help="Sampling step (minutes)")

    ap.add_argument("--height-m", type=float, default=1.5, help="Stake/textile height (m)")
    ap.add_argument("--pressure-hpa", type=float, default=1010.0, help="Pressure (hPa)")
    ap.add_argument("--temp-c", type=float, default=20.0, help="Air temperature (°C)")
    ap.add_argument("--temp-grad", type=float, default=-0.1, help="Near-surface dT/dz (K per m)")
    ap.add_argument("--mirage-layer-m", type=float, default=2.0, help="Thermal layer thickness (m)")
    ap.add_argument("--mirage-coeff", type=float, default=0.02, help="deg per K")
    ap.add_argument("--baseline-m", type=float, default=1000.0, help="Baseline length over which curvature tilt is measured (m)")
    ap.add_argument("--radius-m", type=float, default=6371000.0, help="Effective curvature radius (m)")

    ap.add_argument("--out-csv", type=str, default="shadow_sim.csv", help="Output CSV")
    ap.add_argument("--out-png", type=str, default="shadow_sim.png", help="Output PNG prefix (creates *_sun.png, *_alpha.png, *_shadow.png)")

    args = ap.parse_args()

    day = date.fromisoformat(args.date)
    inputs = ShadowInputs(
        height_m=args.height_m,
        pressure_hPa=args.pressure_hpa,
        temp_C=args.temp_c,
        temp_gradient_K_per_m=args.temp_grad,
        mirage_layer_m=args.mirage_layer_m,
        mirage_coeff_deg_per_K=args.mirage_coeff,
        baseline_m=args.baseline_m,
        radius_m=args.radius_m
    )
    samples = simulate_day(
        lat_deg=args.lat, lon_deg=args.lon, day=day,
        hours_from=args.from_hour, hours_to=args.to_hour, step_min=args.step_min,
        inputs=inputs
    )
    write_csv(samples, args.out_csv)
    try:
        _plot(samples, args.out_png)
    except Exception as e:
        print("Plotting skipped:", e)
    print(f"Wrote {len(samples)} samples to {args.out_csv}")

if __name__ == "__main__":
    main()
