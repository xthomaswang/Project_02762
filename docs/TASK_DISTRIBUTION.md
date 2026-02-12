# Task Distribution

## Team Roles Overview

| Member | Background | Primary Role | Key Folders |
|--------|-----------|--------------|-------------|
| **T** | CS + Robotics | System Integration + Robot Control | `robot/`, `integration/`, `scripts/` |
| **A** | CS + ML | Computer Vision + ML Pipeline | `vision/`, `ml/` |
| **Y** | Biology + Documentation | Experiment Design + Documentation | `experiments/`, `docs/` |
| **D** | Biology + Basic Coding | Data Processing + Experiment Support | `data/`, `scripts/`, `notebooks/` |

---

## Detailed Task Breakdown

### T — System Integration + Robot Control

**Primary folders:** `robot/`, `integration/`, `scripts/`

| Task | Description | Folder | Priority |
|------|-------------|--------|----------|
| OT-2 protocol development | Write Opentrons Python protocols for aspirating/dispensing R/G/B dyes into 96-well plate | `robot/protocols/` | High |
| Robot configuration | Define labware positions, tip rack setup, pipette calibration configs | `robot/configs/` | High |
| End-to-end pipeline | Build the main loop: ML suggests volumes → robot dispenses → camera captures → CV extracts color → ML updates | `integration/` | High |
| Hardware interface | Abstract robot commands so ML/CV modules can call them without OT-2 specifics | `robot/` | Medium |
| Setup scripts | Environment setup, dependency installation, hardware check scripts | `scripts/` | Medium |

**Collaboration:** Works closely with **A** on the integration layer — T handles robot I/O, A handles vision/ML I/O, both co-own `integration/`.

---

### A — Computer Vision + ML Pipeline

**Primary folders:** `vision/`, `ml/`

| Task | Description | Folder | Priority |
|------|-------------|--------|----------|
| Image preprocessing | Crop well plate region, normalize lighting, white balance correction | `vision/preprocessing/` | High |
| Well detection | Detect and locate individual wells in the plate image (circle detection, grid alignment) | `vision/detection/` | High |
| Color extraction | Extract average RGB/HSV values from each detected well | `vision/color_extraction/` | High |
| Surrogate model | Implement Gaussian Process (or other) model mapping dye volumes → observed color | `ml/models/` | High |
| Acquisition function | Implement active learning acquisition (Expected Improvement, UCB, etc.) to suggest next experiments | `ml/acquisition/` | High |
| ML utilities | Color distance metrics (Delta-E), model evaluation, hyperparameter tuning | `ml/utils/` | Medium |

**Collaboration:** Works closely with **T** on integration. Consults **Y** for color science and experiment constraints.

---

### Y — Experiment Design + Documentation

**Primary folders:** `experiments/`, `docs/`

| Task | Description | Folder | Priority |
|------|-------------|--------|----------|
| Experiment design | Define dye concentration ranges, mixing ratios, well plate layout, target colors | `experiments/configs/` | High |
| Experimental protocol | Document step-by-step lab procedures (safety, setup, execution) | `docs/` | High |
| Results analysis | Interpret experimental results, assess color accuracy, convergence analysis | `experiments/results/` | Medium |
| Project report | Write final report: introduction, methods, results, discussion | `docs/` | Medium |
| Progress documentation | Meeting notes, weekly progress, milestone tracking | `docs/` | Ongoing |

**Collaboration:** Works with **D** on experiment parameter design. Provides biological/chemistry context to **T** and **A** for implementation decisions.

---

### D — Data Processing + Experiment Support

**Primary folders:** `data/`, `scripts/`, `notebooks/`

| Task | Description | Folder | Priority |
|------|-------------|--------|----------|
| Synthetic data generation | Generate simulated color mixing data for testing ML pipeline before wet lab | `data/synthetic/`, `scripts/` | High |
| Data preprocessing | Clean and format raw image data and robot logs into structured datasets | `data/processed/` | High |
| Data pipeline scripts | Scripts for data format conversion, batch processing, CSV/JSON export | `scripts/` | Medium |
| Visualization notebooks | Jupyter notebooks for EDA, color space visualization, model performance plots | `notebooks/` | Medium |
| Experiment support | Assist **Y** with experiment parameter tuning based on data analysis | `data/`, `experiments/` | Ongoing |

**Collaboration:** Supports **Y** on experiment design with data-driven insights. Provides processed data to **A** for model training.

---

## Collaboration Map

```
    T (Robot/Integration)  <------>  A (Vision/ML)
           |                              |
           |         integration/         |
           |______________________________|
           |                              |
           v                              v
    Y (Experiment/Docs)   <------>  D (Data/Scripts)
```

- **T ↔ A**: Co-own `integration/`. T handles robot I/O, A handles vision/ML I/O.
- **Y ↔ D**: Co-own experiment design. Y provides domain knowledge, D provides data analysis.
- **T ↔ Y**: Y defines experiment parameters, T translates them into robot protocols.
- **A ↔ D**: D generates synthetic/processed data, A uses it for model development.

---

## Development Phases

### Phase 1: Foundation (Week 1-2)
- [ ] **T**: Set up OT-2 protocol skeleton, test basic liquid handling
- [ ] **A**: Build image preprocessing + well detection pipeline
- [ ] **Y**: Define initial experiment parameters and target colors
- [ ] **D**: Generate synthetic color mixing dataset

### Phase 2: Core Pipeline (Week 3-4)
- [ ] **T**: Complete robot protocols for variable-volume dispensing
- [ ] **A**: Implement surrogate model + acquisition function
- [ ] **Y**: Finalize well plate layout and experiment protocol
- [ ] **D**: Build data preprocessing pipeline for real images

### Phase 3: Integration (Week 5-6)
- [ ] **T + A**: Integrate end-to-end loop (robot → camera → CV → ML → robot)
- [ ] **Y**: Run pilot experiments, document procedures
- [ ] **D**: Validate data pipeline with real experimental data

### Phase 4: Optimization & Report (Week 7-8)
- [ ] **All**: Run full active learning experiments
- [ ] **A**: Tune ML model, compare acquisition strategies
- [ ] **Y + D**: Analyze results, create visualizations
- [ ] **Y**: Write final report
- [ ] **All**: Code cleanup, documentation, final presentation
