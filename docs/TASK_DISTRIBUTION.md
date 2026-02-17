# Task Distribution

## Project Summary

**Title:** Active Learning for Autonomous Colorimetric Assay Design using RGB Dye Mixtures and Computer Vision Feedback

**Objective:** Build an autonomous system that uses an OT-2 liquid-handling robot to mix RGB dyes in a 96-well plate, captures images with a fixed overhead camera, extracts RGB color values via computer vision, and uses Bayesian optimization (Gaussian Process + acquisition functions) to iteratively select the next dye combinations that converge toward a target color.

**One Iteration (per column):**
1. Active learning selects dye volume combination (Vr, Vg, Vb) where Vr + Vg + Vb = 300 µL

   **Phase 1 — Controls (single-channel, left pipette, 4 steps):**
2. Pick up Red tip (A1) → aspirate 300µL Red → dispense into row A → rinse → return tip
3. Pick up Green tip (B1) → aspirate 300µL Green → dispense into row B → rinse → return tip
4. Pick up Blue tip (C1) → aspirate 300µL Blue → dispense into row C → rinse → return tip
5. Pick up Water tip (D1) → aspirate 300µL Water → dispense into row D → rinse → return tip

   **Phase 2 — Experiment (8-channel with 4 tips E-H, right pipette, 4 steps):**
6. Pick up Red tips (col 1, 4 tips) → aspirate Vr → dispense into rows E-H → rinse → return tips
7. Pick up Green tips (col 2, 4 tips) → aspirate Vg → dispense into rows E-H → rinse → return tips
8. Pick up Blue tips (col 3, 4 tips) → aspirate Vb → dispense into rows E-H → rinse → return tips
9. Pick up Mix tips (col 4, 4 tips) → mix wells E-H 3x → rinse → return tips

   **Phase 3 — Image + Learn:**
9. Wait ~45 seconds for settling
11. Camera captures well plate image from fixed overhead USB webcam above slot 5
12. Vision extracts RGB from rows A-D (controls for calibration) and rows E-H (experiment)
13. Average 4 experiment replicates (E-H) → one (R,G,B) measurement per column
13. **3 independent GPs** updated: GP_R, GP_G, GP_B each learn (Vr,Vg,Vb) → channel value
14. Acquisition selects next (Vr,Vg,Vb) minimizing predicted RGB distance to target
15. Repeat — 5 initial + 7 optimization = 12 per plate. If distance > 50, pause for plate/tip swap

**API Approach:** HTTP Robot Server API (not Python Protocol API v2). This is the correct choice because:
- The DL model (YOLO) and active learning run on the external laptop, not on the OT-2's Raspberry Pi
- Real-time camera feedback between robot actions requires external orchestration
- The HTTP API allows sending individual commands (aspirate, dispense, move) with camera checks in between
- The Python Protocol API would require the entire protocol to be uploaded and run on the robot, with no way to inject camera feedback mid-protocol

---

## Hardware Setup

| Component | Details | Location/Notes |
|-----------|---------|----------------|
| **OT-2 Robot** | Liquid handling robot | Room 196 |
| **Left Pipette** | `p300_single_gen2` (20-300 µL, 1-channel) | Left mount — for precise single-well operations |
| **Right Pipette** | `p300_multi_gen2` (20-300 µL, 8-channel) | Right mount — dispenses to 8 wells simultaneously |
| **Camera** | USB camera, fixed mount above OT-2 deck | Fixed position looking down — must not hit glass enclosure |
| **Computer** | Laptop running Python (conda env: `automation`) | Runs vision, ML, and sends HTTP commands to OT-2 |
| **Dyes** | Red, Green, Blue food-grade dyes diluted in **water** | Water diluent — standard viscosity for accurate pipetting |

---

## Deck Layout & Labware Configuration

```
         Left       Center      Right
Back:   [10 empty] [11 empty]  [Trash]
        [ 7 empty] [ 8 empty]  [ 9 empty]
        [ 4 dyes ] [ 5 PLATE ] [ 6 clean]
Front:  [ 1 tips8] [ 2 tips1]  [ 3 empty]
```

| Slot | Labware | API Load Name | Purpose |
|------|---------|---------------|---------|
| **1** | Tiprack 300µL (modified) | `opentrons_96_tiprack_300ul` | 8-ch tips — **rows A,B,C removed!** Only D-H tips remain. Col 1=Red, Col 2=Green, Col 3=Blue, Col 4=Mix |
| **2** | Tiprack 300µL | `opentrons_96_tiprack_300ul` | Single-ch control tips: A1=Red ctrl, B1=Green ctrl, C1=Blue ctrl |
| **4** | NEST 12-Well Reservoir 15mL | `nest_12_reservoir_15ml` | Pre-mixed dye sources: col A1=Red, A2=Green, A3=Blue |
| **5** | **Corning 96-Well Plate 360µL Flat** | `corning_96_wellplate_360ul_flat` | **Dispense plate** — center of deck, camera above |
| **6** | NEST 12-Well Reservoir 15mL | `nest_12_reservoir_15ml` | **Cleaning water** — col A1-A5 for tip rinsing |
| **3, 7-11** | *(empty)* | — | Reserve for future use |
| **12** | Fixed Trash | — | Built-in |

**Well plate layout (every column, all 12):**
```
Row A: Pure Red control    (300µL, single-channel left)  ─┐
Row B: Pure Green control  (300µL, single-channel left)   ├─ Camera calibration
Row C: Pure Blue control   (300µL, single-channel left)   │   (R, G, B references
Row D: Pure Water control  (300µL, single-channel left)  ─┘    + white baseline)
Row E: Experimental mix    (Vr+Vg+Vb, 8-channel right)  ─┐
Row F: Experimental mix    (replicate)                     ├─ GP training data
Row G: Experimental mix    (replicate)                     │   (average 4 wells)
Row H: Experimental mix    (replicate)                    ─┘
```

**Tip-reuse protocol:**
- **8-channel (slot 1):** Tips physically removed from rows A, B, C, D → only **4 tips (E-H)** per column. Dedicated columns: 1=Red, 2=Green, 3=Blue, 4=Mix. Reused with cleaning across all 12 experiments
- **Single-channel (slot 2):** 4 dedicated control tips at A1, B1, C1, D1 (Red, Green, Blue, Water). Each reused with cleaning across all 12 columns
- After each transfer: rinse 3x in cleaning reservoir → blow out → return tips to home position
- **Total tips used: 4×4 (8-ch) + 4 (single-ch) = 20 tips for the entire experiment!**

**Why `corning_96_wellplate_360ul_flat`?**
- Flat-bottom wells give uniform color viewed from above (critical for camera RGB extraction)
- 360µL capacity fits 300µL total volume per well
- Clear wells let camera read colors through the liquid

**Why `nest_12_reservoir_15ml` (×2)?**
- **Dye reservoir (slot 4):** 3 columns of pre-mixed dye (R, G, B), each 15mL
- **Cleaning reservoir (slot 6):** 5 columns of pure water (one per tip set: 4 for 8-ch + 1 for single-ch controls)

**Why controls in every column?**
- **Camera calibration:** Known pure-dye wells (R, G, B) normalize RGB readings for lighting drift
- **White baseline:** Water control (row D) establishes the "zero color" reference point
- **Dye validation:** Verify dye concentration is consistent across the plate
- **No columns wasted:** All 12 columns have both controls AND experiments

---

## Team Roles Overview

| Member | Background | Primary Role | Key Files |
|--------|-----------|--------------|-----------|
| **T (Thomas)** | CS + Robotics | System Integration + Robot Control | `src/robot.py`, `src/protocol.py`, `src/recovery.py`, `scripts/` |
| **A (Alis)** | CS + ML | Computer Vision + ML Pipeline | `src/vision.py`, `src/ml.py`, `src/pipeline.py` |
| **Y (Yeritmary)** | Biology + Documentation | Experiment Design + Documentation | `configs/`, `docs/` |
| **D (David)** | Biology + Basic Coding | Data Processing + Experiment Support | `data/`, `notebooks/` |

---

## Detailed Task Breakdown

### T (Thomas) — System Integration + Robot Control

**Primary files:** `src/robot.py`, `src/protocol.py`, `src/recovery.py`

| # | Task | Description | File | Status |
|---|------|-------------|------|--------|
| T1 | OT-2 robot control | `OT2Robot` class wrapping HTTP API commands (aspirate, dispense, move, pick_up, drop_tips, blow_out, home) | `src/robot.py` | Done |
| T2 | Protocol execution | YAML-driven task executor with vision verification at each step (pickup → check tips, transfer → check liquid level) | `src/protocol.py` | Done |
| T3 | Error recovery | Config-driven recovery plans (retry, home, pause) with runtime context substitution | `src/recovery.py` | Done |
| T4 | Add left pipette support | Update `robot.py` and `protocol.py` to support both left (`p300_single_gen2`) and right (`p300_multi_gen2`) pipettes. Single-channel can be used for precise per-well dispensing when 8-channel is too coarse | `src/robot.py`, `src/protocol.py` | **High** |
| T5 | 7-step dispensing protocol | Implement per-column sequence: **Phase 1 (controls, left pipette):** 3 single-channel transfers of pure R/G/B into rows A/B/C with tip reuse+rinse. **Phase 2 (experiment, right 8-ch with 5 tips):** 3 dye transfers (R→G→B into rows D-H) + 1 mix step, each with tip reuse+rinse. Volumes from ML. Vr+Vg+Vb=300µL | `src/protocol.py` | **High** |
| T6 | Integrate robot into pipeline | Connect `_run_single_experiment()` to 7-step protocol. Handle: dual-pipette coordination (left for controls, right for experiments), tip reuse with return-to-home, cleaning between uses, column progression (1→12), batch mode pause after 12 | `src/pipeline.py` | **High** |
| T7 | Connection resilience | Add retry logic for transient HTTP failures, connection timeout handling, and automatic reconnection to OT-2 | `src/robot.py` | Medium |
| T8 | Entry point scripts | CLI scripts with argument parsing for running experiments. Already functional, may need updates for new config fields | `scripts/run_experiment.py` | Low (mostly done) |

**Technical notes for T:**
- **Dual-pipette coordination:** Left single-channel does controls (rows A-C), then right 8-channel does experiments (rows D-H). Controls first to avoid splash contamination
- **Modified tiprack (slot 1):** Physically remove tips at rows A, B, C, D from columns 1-4. The 8-channel picks up 4 tips (E-H) and skips empty positions A-D
- **Tip reuse with cleaning:** After each transfer: (1) move to cleaning reservoir, (2) aspirate 250µL, (3) dispense 250µL, (4) repeat 3x, (5) blow out, (6) return tips to home. Same tips for all 12 columns
- **Volume constraint:** Vr + Vg + Vb = 300µL. Pre-mixed dyes, no separate water
- **Mix step:** 8-channel aspirate 200µL from wells D-H, dispense back, repeat 3x
- **Column progression:** 1→12. After 12, pause for plate swap + refill cleaning water. Tips reusable across batches
- HTTP API endpoint: `http://{ip}:{port}/runs/{run_id}/commands` with POST method

---

### A (Alis) — Computer Vision + ML Pipeline

**Primary files:** `src/vision.py`, `src/ml.py`, `src/pipeline.py`

| # | Task | Description | File | Status |
|---|------|-------------|------|--------|
| A1 | Camera + YOLO detection | `Camera` class, `Predict()`, `check_tip_with_model()`, `check_liquid_level()` | `src/vision.py` | Done |
| A2 | GP surrogate model | **Refactor to 3 independent GPs** (GP_R, GP_G, GP_B). Each maps (Vr,Vg,Vb) → one RGB channel. Current SingleTaskGP code can be reused 3x. The 8 replicate wells per column should be averaged to get one (R,G,B) observation per experiment | `src/ml.py` | **High** (refactor needed) |
| A3 | Acquisition functions | **Refactor for multi-output.** Acquisition should minimize predicted RGB distance: `sqrt((GP_R(V)-target_R)² + (GP_G(V)-target_G)² + (GP_B(V)-target_B)²)`. Use composite EI or optimize the predicted distance directly. Must also enforce constraint Vr+Vg+Vb=300 | `src/ml.py` | **High** (refactor needed) |
| A4 | **Color extraction from wells** | **CRITICAL — currently missing.** Implement `extract_well_colors(image_path, well_positions) -> np.ndarray` that: (1) loads captured plate image, (2) detects/locates wells using HoughCircles or grid-based detection, (3) extracts mean RGB from center region of each well (avoid edges/reflections), (4) returns array of shape (n_wells, 3). Consider using a circular ROI ~60-70% of well diameter to avoid edge artifacts | `src/vision.py` | **Critical** |
| A5 | **Complete active learning loop** | **CRITICAL — `_run_single_experiment()` is currently a placeholder.** Must implement: (1) call 4-step dispensing protocol (R→G→B→Mix), (2) `time.sleep(45)` for settling, (3) capture image with Camera, (4) extract RGB from 8 wells in column, (5) average 8 replicates → one (R,G,B), (6) update 3 GPs, (7) acquisition suggests next (Vr,Vg,Vb). Add convergence check: if distance < 50, stop early | `src/pipeline.py` | **Critical** |
| A6 | Camera calibration for color | Develop white-balance / lighting normalization so RGB values are consistent across sessions. Options: include a white reference well, apply color correction matrix, or normalize against a known color standard | `src/vision.py` | **High** |
| A7 | Well plate grid mapping | Map pixel coordinates to well IDs (A1-H12). For fixed camera position, this can be a one-time calibration (store as config). For the 8-channel, you mainly need column-level mapping | `src/vision.py`, `configs/` | **High** |
| A8 | Color distance metric | `color_distance()` computes Euclidean RGB distance from the 3 GP predictions vs target. Consider also CIELAB (`deltaE`) for perceptually uniform comparison. Convergence threshold: distance < 50 | `src/ml.py` | Medium |
| A9 | Convergence visualization | Plot optimization progress: best distance vs iteration, GP prediction surface, acquisition function landscape | `src/ml.py` or `notebooks/` | Medium |

**Technical notes for A:**
- **Pure RGB approach:** 3 independent GPs (one per channel) instead of 1 GP on scalar distance. This preserves directional information (the model knows "too much red" vs "not enough blue"), leading to faster convergence with limited data (12 experiments)
- **4 replicates per experiment:** Each column has 4 experiment wells (E-H) with identical mix. Average: `mean_rgb = np.mean(well_rgbs, axis=0)`. Report std across 4 wells as noise estimate for GP. Rows A-D are controls (R, G, B, Water) for camera calibration
- **Volume constraint:** Vr + Vg + Vb = 300µL. This makes the search space effectively 2D (Vb = 300 - Vr - Vg). The acquisition function must enforce this constraint when suggesting new volumes
- **Well detection approach:** For a fixed overhead USB webcam, use OpenCV `cv2.HoughCircles()` on grayscale image to detect circular wells. Since the plate is always in slot 5, define a grid template once and reuse it
- **RGB extraction:** After isolating each well ROI, compute `np.mean(roi, axis=(0,1))` over the center ~60% of the circle to get mean RGB. Avoid the well rim (white plastic) which skews values
- **Lighting:** Consistent lighting is critical. Consider a white LED panel or diffuser to reduce ambient light variability

---

### Y (Yeritmary) — Experiment Design + Documentation

**Primary files:** `configs/`, `docs/`

| # | Task | Description | Location | Status |
|---|------|-------------|----------|--------|
| Y1 | Experiment config | Define target colors, dye concentration ranges, well plate layout, and dispensing parameters in YAML | `configs/experiment.yaml` | **High** |
| Y2 | Target color selection | Choose 3-5 target RGB colors of varying difficulty for the active learning to converge toward. Include easy targets (single dye) and hard targets (3-dye mixtures). Document rationale | `configs/`, `docs/` | **High** |
| Y3 | Dye preparation + pilot test | **Run pilot test first:** manually prepare 3-4 dilution ratios (1:5, 1:10, 1:20 dye:water), dispense 300µL of each into a well, photograph, and pick the dilution where colors are vivid but not saturated. Document the chosen concentration and preparation protocol | `docs/` | **High** |
| Y4 | Lab safety & setup | Document lab safety for room 196: PPE requirements, dye handling, OT-2 startup/shutdown, camera positioning, cleanup procedures | `docs/` | **High** |
| Y5 | Well plate layout design | Design which columns get which experiments. With 8-channel (12 columns available): allocate columns for replicates, controls (water-only, single-dye), and active learning experiments | `configs/`, `docs/` | **High** |
| Y6 | Request 8-channel mount | The Materials.docx notes: "need to request an 8-channel mount for OT-2 in room 196". Ensure this is done and the mount is installed | Physical task | **High** |
| Y7 | Camera mount solution | Materials.docx mentions "can we 3D print an attachment? Or can we use tape?" — for fixed overhead mount, a simple clamp or 3D-printed bracket attached to the OT-2 frame is most reliable. Camera must not hit the glass enclosure | Physical task | **High** |
| Y8 | Pilot experiment documentation | Run 1-2 manual pilot experiments, document observations: dye mixing time, color stability, image quality, any issues | `docs/` | Medium |
| Y9 | Results interpretation | After active learning runs, analyze convergence: how many iterations to reach target color within threshold? Which acquisition function (EI vs UCB) performed better? | `notebooks/`, `docs/` | Medium |
| Y10 | Final project report | Report with: introduction, methods (hardware, software, ML), results, discussion, future work | `docs/` | Medium |

**Technical notes for Y:**
- **Dye concentration matters:** If dyes are too concentrated, small volume differences produce big color changes (hard to optimize). If too dilute, colors are faint and hard to measure. Aim for concentrations where 0-300µL volume range covers most of the RGB gamut
- **Mixing time:** After dispensing, dyes need time to mix in the well. With water (low viscosity), gentle dispensing from the 8-channel creates enough turbulence. Allow ~30-60 seconds settling time before imaging
- **Controls:** Include at least one column with known volumes (e.g., equal parts R+G+B) as a reference to validate the vision system between runs

---

### D (David) — Data Processing + Experiment Support

**Primary files:** `data/`, `notebooks/`

| # | Task | Description | Location | Status |
|---|------|-------------|----------|--------|
| D1 | Synthetic color mixing data | Generate simulated RGB data: given (Vr, Vg, Vb) volumes, predict resulting RGB color using a physically-informed model. Use Beer-Lambert-like attenuation per channel. Add Gaussian noise to simulate camera variability. Minimum 200-500 samples | `data/synthetic/`, `notebooks/` | **High** |
| D2 | Synthetic data model | Implement `simulate_color_mixing(Vr, Vg, Vb) -> (R, G, B)`. Consider: each dye primarily affects one RGB channel; higher volume = more saturated color; total volume normalization | `notebooks/` | **High** |
| D3 | Offline ML validation | Use synthetic data to test the full GP + acquisition loop before running on real hardware. Verify that: (a) GP fits the data correctly, (b) acquisition function selects informative points, (c) loop converges toward target | `notebooks/` | **High** |
| D4 | Data preprocessing pipeline | For real experiment data: organize images by run/iteration, extract metadata (timestamp, volumes, well IDs), store in structured format (CSV or JSON) | `data/`, `notebooks/` | **High** |
| D5 | Image quality checks | Build notebook to verify captured images: correct framing, consistent lighting, well detection accuracy. Compare RGB extracted from known dye concentrations vs expected values | `notebooks/` | Medium |
| D6 | Convergence visualization | Jupyter notebooks showing: optimization trajectory, GP prediction heatmaps, acquisition function landscape, best-so-far curves, color swatches (observed vs target) | `notebooks/` | Medium |
| D7 | Statistical analysis | After experiments: compute convergence rates, compare EI vs UCB, analyze reproducibility across runs, report confidence intervals | `notebooks/` | Medium |
| D8 | Experiment logging support | Assist T in designing the experiment log format: every iteration should save (volumes, image_path, extracted_RGB, color_distance, GP_prediction, timestamp) | `data/`, `src/pipeline.py` | Medium |

**Technical notes for D:**
- **Synthetic color model suggestion:** `RGB_observed = RGB_background * exp(-k * V / V_max)` where k is channel-specific attenuation. Red dye: k_R=3, k_G=0.5, k_B=0.5. Add noise: `+ N(0, σ=5)` per channel
- **Data directory structure suggestion:**
  ```
  data/
  ├── synthetic/          # Generated offline data
  │   ├── training.csv    # (Vr, Vg, Vb, R, G, B)
  │   └── generation.ipynb
  ├── runs/               # Real experiment data
  │   ├── run_001/
  │   │   ├── metadata.json
  │   │   ├── images/
  │   │   └── results.csv
  │   └── run_002/
  └── calibration/        # Camera calibration images
  ```

---

## Collaboration Map

```
    T (Robot/Protocol)  ←————————→  A (Vision/ML)
         |                               |
         |       src/pipeline.py         |
         |   (T handles robot calls,     |
         |    A handles vision+ML)       |
         |_______________________________|
         |                               |
         v                               v
    Y (Experiment/Docs)  ←————————→  D (Data/Notebooks)
         |                               |
         | configs/experiment.yaml       | data/, notebooks/
         | (Y defines what to test,      | (D processes results,
         |  parameters, protocols)       |  validates pipeline)
```

**Key Integration Points:**
- **T ↔ A:** Both contribute to `src/pipeline.py`. T writes robot command sequences, A writes vision+ML logic. Must agree on function signatures
- **T ↔ Y:** Y defines experiment parameters in `configs/experiment.yaml`, T implements them in protocol execution
- **A ↔ D:** D provides synthetic data for A to validate ML pipeline offline before real experiments
- **Y ↔ D:** D analyzes results that Y interprets for the report

---

## Issues Found & Corrections

### From Materials.docx
| Issue | Status | Action Required |
|-------|--------|-----------------|
| "Need to request 8-channel mount for OT-2 in room 196" | **Blocking** | Y must confirm this is done before Phase 2 testing |
| "Camera — need to check box in 262 if any issues arise" | Open | Verify camera model and resolution are sufficient |
| "Camera mount — 3D print or tape?" | Open | Recommend 3D-printed bracket for stability. Tape is unreliable and shifts during operation |
| "Camera not hitting glass on robot" | Constraint | Fixed overhead mount avoids this issue entirely |
| "Check literature for standardizing absorbance spectrum of RGB colors" | Clarified | We are NOT measuring absorbance — we measure RGB color distance with camera. Terminology in proposal should say "RGB color distance" not "absorbance spectrum" |
| "RGB 1:1 controlled dye mixtures with water or glycerol?" | **Decided: Water** | Water is standard for OT-2 pipetting calibration. Glycerol's high viscosity causes inaccurate dispensing |

### From Project Proposal
| Issue | Correction |
|-------|------------|
| Repeated use of "absorbance spectrum" | Should be "RGB color distance" — we use camera, not spectrophotometer |
| "Deep Learning Model to read RGB values" | Current implementation uses OpenCV for RGB extraction (simpler, sufficient). YOLO is used only for tip/liquid detection, not color reading |
| "minimum of 5 cycles" | 5 is absolute minimum. Recommend 15-30 iterations for meaningful convergence with GP + EI |
| "find the spectra optimal conditions" | Not applicable — we optimize dye volumes to match a target RGB, not spectral conditions |

### From Codebase
| Issue | File | Correction |
|-------|------|------------|
| `pipeline.py:_run_single_experiment()` returns random data | `src/pipeline.py` | **Must be implemented** — currently a skeleton placeholder |
| No color extraction function exists | `src/vision.py` | **Must be implemented** — this is the core measurement |
| Config only has right pipette (`p300_multi_gen2`) | `configs/experiment.yaml` | Add left pipette (`p300_single_gen2`) config |
| `imaging` labware is PCR strip, not suitable for colorimetric reading | `configs/experiment.yaml` | Change to appropriate plate or remove if imaging happens on the dispense plate directly |
| `LLD_CALIBRATION_CSV` env var dependency | `src/vision.py` | Should be in config file, not environment variable |
| No experiment state persistence | `src/pipeline.py` | Add saving of (volumes, images, RGB, distances) per iteration |

---

## Updated Config: `configs/experiment.yaml`

The current config needs these updates:

```yaml
# ---- Changes needed ----
# 1. Add left pipette
labware:
  pipette_right:
    name: "p300_multi_gen2"    # 8-channel, right mount
  pipette_left:
    name: "p300_single_gen2"   # 1-channel, left mount
  tiprack_right:
    name: "opentrons_96_tiprack_300ul"
    slot: "1"
  tiprack_left:
    name: "opentrons_96_tiprack_300ul"
    slot: "2"
  sources:
    name: "nest_12_reservoir_15ml"
    slots: ["4"]
    # Column mapping: A1=Red dye, A2=Green dye, A3=Blue dye, A4=Water
  dispense:
    name: "corning_96_wellplate_360ul_flat"
    slot: "7"

# 2. Add camera config
camera:
  device_id: 0
  width: 1920
  height: 1080
  warmup_frames: 10
  mount: "fixed_overhead"

# 3. Add mixing/timing config
mixing:
  settle_time_seconds: 45    # Wait after dispensing before imaging
  total_volume_ul: 300       # Normalize all wells to this total volume with water
```

---

## Development Phases (Updated)

### Phase 1: Foundation (Week 1-2) ✅ Mostly Complete
- [x] **T**: Port OT-2 control code into OT2Robot class (`src/robot.py` — 366 lines)
- [x] **T**: Port protocol execution + error recovery (`src/protocol.py`, `src/recovery.py`)
- [x] **A**: Port vision detection pipeline (`src/vision.py` — camera, YOLO, tip/liquid checks)
- [x] **A**: Implement GP surrogate model + acquisition functions (`src/ml.py` — BoTorch)
- [ ] **Y**: Define initial experiment parameters and target colors → **Update `configs/experiment.yaml` with corrected labware**
- [ ] **Y**: Confirm 8-channel mount availability in room 196
- [ ] **D**: Generate synthetic color mixing dataset (200+ samples of Vr,Vg,Vb → R,G,B)

### Phase 2: Core Pipeline (Week 3-4)
- [ ] **T**: Add left pipette support (`p300_single_gen2`) to robot.py and protocol.py
- [ ] **T**: Implement dynamic volume dispensing (ML-suggested volumes → robot commands)
- [ ] **T**: Test robot commands against live OT-2 (basic aspirate/dispense/move)
- [ ] **A**: **Implement color extraction from well plate images** (HoughCircles + mean RGB)
- [ ] **A**: Implement camera calibration / white-balance normalization
- [ ] **A**: Integrate vision color extraction into `pipeline.py`
- [ ] **Y**: Prepare dye solutions (R, G, B in water at target concentrations)
- [ ] **Y**: Design and document camera mount solution (3D print recommended)
- [ ] **Y**: Finalize well plate layout (which columns for experiments, controls, replicates)
- [ ] **D**: Validate GP + acquisition loop with synthetic data (offline test)
- [ ] **D**: Build data preprocessing pipeline for real images
- [ ] **D**: Design experiment logging format (CSV + image directory structure)

### Phase 3: Integration (Week 5-6)
- [ ] **T + A**: Complete `_run_single_experiment()` in `pipeline.py` with real robot + vision
- [ ] **T + A**: End-to-end test: dispense known volumes → image → extract RGB → verify against expected colors
- [ ] **T**: Add connection resilience (retry, timeout, reconnection)
- [ ] **A**: Test well detection reliability across different plate positions and lighting
- [ ] **Y**: Run 2-3 pilot experiments, document procedures and observations
- [ ] **Y**: Verify dye mixing behavior (settling time, color stability)
- [ ] **D**: Validate data pipeline with real experimental data
- [ ] **D**: Create quality-check notebook for image/color verification

### Phase 4: Optimization & Report (Week 7-8)
- [ ] **All**: Run full active learning experiments (minimum 5 iterations, target 15-30)
- [ ] **A**: Tune GP hyperparameters, compare EI vs UCB acquisition strategies
- [ ] **A**: Implement CIELAB color distance as alternative metric
- [ ] **T**: Optimize dispensing speed and tip management for throughput
- [ ] **Y + D**: Analyze convergence results, create visualizations
- [ ] **D**: Statistical comparison of acquisition strategies
- [ ] **Y**: Write final report (introduction, methods, results, discussion)
- [ ] **All**: Code cleanup, documentation, final presentation

---

## Environment Setup

```bash
# Conda environment (Python 3.10)
conda activate automation    # /Users/tuomasier/miniconda3/envs/automation

# Install dependencies
pip install -r requirements.txt

# Key packages: requests, opencv-python, Pillow, super-gradients,
#   numpy, scipy, scikit-learn, botorch, gpytorch, torch,
#   pandas, matplotlib, seaborn, pyyaml, pytest

# Run experiment
python scripts/run_experiment.py --config configs/experiment.yaml --n-initial 5 --n-iterations 20
```

---

## References

- [Opentrons Labware Library](https://labware.opentrons.com/)
- [Opentrons Python API v2 — Pipette Loading](https://docs.opentrons.com/v2/pipettes/loading.html)
- [Well and Color Detection of PCR Plate using Python and OpenCV](https://medium.com/codex/well-and-color-detection-of-pcr-plate-using-python-and-opencv-edb0aef9d)
- [Benchmarking Bayesian Optimization across Experimental Materials Science Domains (Nature)](https://www.nature.com/articles/s41524-021-00656-9)
- [Exploring Bayesian Optimization (Distill.pub)](https://distill.pub/2020/bayesian-optimization/)
- [Smartphone RGB Camera for Colorimetric Assay (PubMed)](https://pubmed.ncbi.nlm.nih.gov/31514873/)
