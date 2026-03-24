# Study on Nazca Lines

A symbolic-physics framework for shadow projection, curvature detection, and mirage modeling.

## Project Structure

```
shadow/
  __init__.py          # Package init, re-exports public API
  shadow.py            # Core physics: solar position, refraction, mirage, curvature, shadow sim
docs/
  shadow_demo.csv      # Sample output (Nazca, equinox 2025-09-21)
  shadow_demo_nazca_sun.png  # Sun elevation plot
requirements.txt       # Python dependencies
```

## Core Equations

The shadow simulator unifies three physical effects into a single effective solar altitude:

```
α_eff = α_true + Δα_refraction + Δα_mirage − Δα_curvature
```

Then shadow length:

```
L = h / tan(α_eff)       (returns None when α_eff ≤ 0)
```

Where:
- **Refraction** (Bennett 1982): `R ≈ 0.0002967 · P / (273.15 + T) / tan(α + 7.31/(α + 4.4))`
- **Mirage shift**: `Δθ_m = coeff · (dT/dz) · layer_thickness`
- **Curvature tilt**: `Δθ_c = baseline / (2R)` radians — sagitta slope at midspan
- **Solar position**: Approximate NOAA algorithm (Julian day → ecliptic coords → equatorial → horizon)

## Dev Commands

```bash
# Run simulation (Nazca Pampa, equinox)
python -m shadow.shadow \
  --lat -14.735 --lon -75.13 --date 2025-09-21 \
  --from-hour 12 --to-hour 22 --step-min 5 \
  --height-m 1.5 --baseline-m 1000 \
  --out-csv shadow_sim.csv --out-png shadow_sim.png

# Run tests
python -m pytest tests/ -v

# Import as library
python -c "from shadow import solar_position, simulate_day; print('OK')"
```

## Dependencies

- Python >= 3.9
- matplotlib (optional, for plots)

## Key Design Decisions

- Solar altitude corrections are additive: refraction lifts apparent sun, mirage shifts it,
  curvature tilts the ground away — all combined before the single `h/tan(α)` projection.
- The mirage model is a linear first-order approximation. For strong mirages (> 1° shift),
  a ray-tracing approach would be needed.
- `ground_accuracy_m()` converts textile thread spacing to ground-scale projection accuracy
  for correlating geoglyph line widths with weaving resolution.
