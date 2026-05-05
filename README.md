# DSA 210: Introduction to Data Science - Term Project
## 3D Printing Parameter Analysis

**Term:** 2025-2026 Spring  
**Student:** Ahmet Deha Yıldırım  
**Student ID:** 00032656  

---

## Project Overview

This project investigates how 3D printing parameters affect print quality and mechanical performance. The analysis focuses on how controllable printing settings affect three output variables: surface roughness, tensile strength, and elongation.

---

## Dataset

The main dataset contains 50 experimental observations from 3D printing processes using an Ultimaker S5 3D printer (Selçuk University Mechanical Engineering Department).

**Input Variables:**
- `layer_height` — thickness of each printed layer (mm)
- `wall_thickness` — thickness of the outer walls (mm)
- `infill_density` — internal fill density (%)
- `infill_pattern` — internal fill pattern (grid / honeycomb)
- `nozzle_temperature` — printing nozzle temperature (°C)
- `bed_temperature` — print bed temperature (°C)
- `print_speed` — printing speed (mm/s)
- `material` — filament material type (PLA / ABS)
- `fan_speed` — cooling fan speed (%)

**Output Variables:**
- `roughness` — surface roughness (µm)
- `tensile_strength` — ultimate tensile strength (MPa)
- `elongation` — elongation at break (%)

---

## Data Enrichment

Since the original dataset contains only 50 samples, two enrichment steps were applied:

**1. Feature Engineering** — 6 new physically meaningful features were derived:
- `heat_ratio` = nozzle_temperature / bed_temperature
- `thermal_delta` = nozzle_temperature - bed_temperature
- `volumetric_flow` = layer_height × print_speed
- `infill_wall_ratio` = infill_density / wall_thickness
- `material_pla` — binary encoding of material type
- `pattern_honeycomb` — binary encoding of infill pattern

**2. Gaussian Noise Augmentation** — 4 copies of the original data were generated with 3% Gaussian noise added to numerical features, resulting in 250 total samples. All augmented values were clipped to original feature ranges to preserve physical validity.

---

## Hypotheses

| # | Hypothesis | Result |
|---|-----------|--------|
| H1 | Layer height has a significant effect on surface roughness | ✅ Supported — r = 0.802, p < 0.001 |
| H2 | Nozzle temperature has a significant effect on elongation and tensile strength | ✅ Supported — elongation r = -0.526, tensile strength r = -0.402, p < 0.001 |
| H3 | Material type (PLA vs ABS) has a significant effect on all output variables | ✅ Supported — p < 0.05 for all three outputs |
| H4 | Infill pattern (honeycomb vs grid) has a significant effect on mechanical properties | ❌ Rejected — p > 0.05, no significant difference found |
| H5 | Volumetric flow (layer height × print speed) has a significant effect on surface roughness | ✅ Supported — r = 0.641, p < 0.001 |

---

## Machine Learning Results

Four regression models were trained and evaluated for each output variable using 80/20 train-test split and 5-fold cross-validation.

| Target | Best Model | R² |
|--------|-----------|-----|
| roughness | XGBoost | 0.992 |
| tensile_strength | XGBoost | 0.975 |
| elongation | SVR | 0.987 |

**Models compared:** Linear Regression, Random Forest, XGBoost, SVR
