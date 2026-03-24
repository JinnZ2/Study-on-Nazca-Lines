# Study on Nazca Lines

A symbolic-physics framework for shadow projection, curvature detection, and mirage modeling.

## Overview

This framework models how shadows behave under curvature and mirage-like conditions. Unlike flat-plane shadow geometry, it encodes atmospheric, refractive, and curvature variables into formulas that reveal hidden structure in figure bending, displacement, and shadow persistence.

Shadows here are not passive projections. They are informational traces that encode:
- Curvature of the medium (surface or spacetime)
- Mirage conditions (temperature gradients, density changes, refractive layering)
- Observer dependence (angle, distance, perception thresholds)

The goal is to treat shadow as a measurement system — a natural sensor for underlying curvature.

---

## Core Equations

### Unified Effective Solar Altitude

The code combines three corrections into a single effective altitude before projecting the shadow:

$$\alpha_{\text{eff}} = \alpha_{\text{true}} + \Delta\alpha_{\text{refraction}} + \Delta\alpha_{\text{mirage}} - \Delta\alpha_{\text{curvature}}$$

Shadow length is then:

$$L = \frac{h}{\tan(\alpha_{\text{eff}})}$$

Returns no intersection when $\alpha_{\text{eff}} \le 0$ (sun at or below effective horizon).

### 1. Atmospheric Refraction (Bennett 1982)

$$\Delta\alpha_R = \frac{0.0002967 \cdot P}{(273.15 + T) \cdot \tan\!\left(\alpha + \frac{7.31}{\alpha + 4.4}\right)}$$

- $P$ = atmospheric pressure (hPa)
- $T$ = air temperature (C)
- $\alpha$ = apparent solar altitude (deg)

### 2. Mirage Angular Shift

$$\Delta\theta_m = c \cdot \frac{dT}{dz} \cdot \Delta z$$

- $c$ = empirical coefficient (deg/K, default 0.02)
- $dT/dz$ = near-surface temperature gradient (K/m)
- $\Delta z$ = thermal layer thickness (m)

Negative gradient (hot surface) produces inferior mirage (apparent lowering).

### 3. Curvature Tilt

$$\Delta\theta_c = \frac{b}{2R}$$

- $b$ = baseline length (m) over which curvature is measured
- $R$ = effective curvature radius (m, default Earth radius 6,371 km)

This is the sagitta-slope approximation at midspan.

### 4. Mirage Threshold Condition (conceptual, not yet implemented)

$$\Delta n(z) > \nabla T(z) \cdot \alpha$$

When true, shadow behaves non-linearly (floating, displaced, inverted).

---

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run simulation (Nazca Pampa, equinox)
python -m shadow.shadow \
  --lat -14.735 --lon -75.13 --date 2025-09-21 \
  --from-hour 12 --to-hour 22 --step-min 5 \
  --height-m 1.5 --out-csv shadow_sim.csv --out-png shadow_sim.png

# Use as library
from shadow import solar_position, simulate_day, ShadowInputs
```

---

## Applications

- Curvature detection — use shadows as natural rulers of planetary or local curvature
- Mirage modeling — predict when shadow anomalies arise under desert, ice, or water-layered conditions
- Geometric inversion — reconstruct hidden curvature by analyzing shadow displacement across conditions
- Textile-to-ground projection — correlate geoglyph line widths with weaving thread spacing via `ground_accuracy_m()`

---

## Symbolic Integration

Shadow is treated as a mirror of form in the symbolic intelligence system:
- `Shadow = FORM x (LIGHT x MEDIUM)`
- Curvature enters as hidden gradient
- Mirage enters as transformation operator

---

## Roadmap

- [x] Python module for simulation (`shadow.py`)
- [ ] Implement mirage threshold condition (equation 4)
- [ ] Integrate with symbolic sensor suite
- [ ] Visualization of curvature vs shadow displacement
- [ ] Nighttime/moonlight diffraction extension
- [ ] Architectural and plasma-shadow analogies

---

## License

Open-source under MIT.
