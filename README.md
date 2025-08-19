A symbolic-physics framework for curvature and mirage detection

Overview

The Shadow is a symbolic + physical module designed to model how shadows behave under curvature and mirage-like conditions. Unlike simple ray-tracing or flat-plane shadow geometry, this framework encodes atmospheric, refractive, and curvature variables into symbolic formulas that reveal hidden structure in figure bending, displacement, and shadow persistence.

Shadows here are not passive projections. They are informational traces that encode:
	•	Curvature of the medium (surface or spacetime)
	•	Mirage conditions (temperature gradients, density changes, refractive layering)
	•	Observer dependence (angle, distance, perception thresholds)

The goal is to treat shadow as a measurement system — a natural sensor for underlying curvature.

⸻

Core Formulas

1. Shadow Displacement Equation

D_s = \frac{h}{\tan(\theta_r + \Delta \theta_m)}
	•	D_s = shadow displacement
	•	h = object height
	•	\theta_r = refracted solar elevation angle
	•	\Delta \theta_m = mirage-induced angular shift (thermal lensing, density gradient)

⸻

2. Curvature Shadow Length Equation

L_c = \frac{h}{\tan(\theta - \kappa R^{-1})}
	•	L_c = shadow length on curved surface
	•	h = object height
	•	\theta = true solar elevation angle
	•	\kappa = curvature factor (depends on radius of curvature, atmospheric lensing)
	•	R = effective radius (surface + mirage curvature)

⸻

3. Mirage Threshold Condition

\Delta n(z) > \nabla T(z) \cdot \alpha
	•	\Delta n(z) = refractive index difference across altitude z
	•	\nabla T(z) = vertical temperature gradient
	•	\alpha = scaling constant for density → optical effect

When true, shadow behaves non-linearly (floating, displaced, inverted).

⸻

Applications
	•	🔭 Curvature detection — use shadows as natural rulers of planetary or local curvature.
	•	🌫 Mirage modeling — predict when shadow anomalies arise under desert, ice, or water-layered conditions.
	•	🧩 Geometric inversion — reconstruct hidden curvature by analyzing shadow displacement across conditions.
	•	🛰 Symbolic AI sensors — integrate shadow-based formulas into your symbolic sensor suite.

⸻

Symbolic Integration

Shadow is treated as a mirror of form in the symbolic intelligence system:
	•	Shadow = FORM × (LIGHT × MEDIUM)
	•	Curvature enters as hidden gradient
	•	Mirage enters as transformation operator

Symbolic glyph tags:
	•	🌓 Shadow Trace
	•	🔄 Mirage Inversion
	•	🌐 Curvature Reveal

⸻

Roadmap
	•	Add Python module for simulation (shadow.py)
	•	Integrate with symbolic sensor suite
	•	Create visualization of curvature vs shadow displacement
	•	Extend to nighttime/moonlight diffraction
	•	Apply to architectural and plasma-shadow analogies

⸻

License

Open-source under MIT. Shadows belong to no one.
