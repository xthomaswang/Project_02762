# Task Distribution

## Team Roles Overview

| Member | Background | Primary Role | Key Files |
|--------|-----------|--------------|-----------|
| **T** | CS + Robotics | System Integration + Robot Control | `src/robot.py`, `src/protocol.py`, `src/recovery.py`, `scripts/` |
| **A** | CS + ML | Computer Vision + ML Pipeline | `src/vision.py`, `src/ml.py`, `src/pipeline.py` |
| **Y** | Biology + Documentation | Experiment Design + Documentation | `configs/`, `docs/` |
| **D** | Biology + Basic Coding | Data Processing + Experiment Support | `data/`, `notebooks/` |

---

## Detailed Task Breakdown

### T -- System Integration + Robot Control

**Primary files:** `src/robot.py`, `src/protocol.py`, `src/recovery.py`

| Task | Description | File | Priority |
|------|-------------|------|----------|
| OT-2 robot control | OT2Robot class for HTTP API commands (aspirate, dispense, move, etc.) | `src/robot.py` | Done |
| Protocol execution | Task-based protocol executor with vision verification | `src/protocol.py` | Done |
| Error recovery | Config-driven recovery plan execution | `src/recovery.py` | Done |
| End-to-end pipeline | Integrate robot actions into active learning loop | `src/pipeline.py` | High |
| Entry point scripts | CLI scripts for running experiments | `scripts/` | Medium |

---

### A -- Computer Vision + ML Pipeline

**Primary files:** `src/vision.py`, `src/ml.py`, `src/pipeline.py`

| Task | Description | File | Priority |
|------|-------------|------|----------|
| Camera + detection | Camera class, YOLO prediction, tip/liquid checks | `src/vision.py` | Done |
| Surrogate model | Gaussian Process model mapping dye volumes to observed color | `src/ml.py` | Done |
| Acquisition function | Expected Improvement / UCB for next-experiment selection | `src/ml.py` | Done |
| Active learning loop | Orchestrate ML + robot + vision in iterative loop | `src/pipeline.py` | High |
| Color extraction | Extract RGB from well plate images (integrate into vision.py) | `src/vision.py` | High |

---

### Y -- Experiment Design + Documentation

**Primary files:** `configs/`, `docs/`

| Task | Description | Location | Priority |
|------|-------------|----------|----------|
| Experiment config | Define dye ranges, target colors, well plate layout | `configs/experiment.yaml` | High |
| Lab procedures | Document safety, setup, and execution protocols | `docs/` | High |
| Results analysis | Interpret results, assess convergence | `notebooks/` | Medium |
| Project report | Final report with methods, results, discussion | `docs/` | Medium |

---

### D -- Data Processing + Experiment Support

**Primary files:** `data/`, `notebooks/`

| Task | Description | Location | Priority |
|------|-------------|----------|----------|
| Synthetic data | Simulated color mixing data for offline ML testing | `data/`, `notebooks/` | High |
| Data preprocessing | Clean/format raw images and robot logs | `data/` | High |
| Visualization | Jupyter notebooks for EDA and model performance | `notebooks/` | Medium |
| Experiment support | Assist Y with parameter tuning via data analysis | `notebooks/` | Ongoing |

---

## Collaboration Map

```
    T (Robot/Protocol)  <------>  A (Vision/ML)
           |                           |
           |      src/pipeline.py      |
           |___________________________|
           |                           |
           v                           v
    Y (Experiment/Docs)  <------>  D (Data/Notebooks)
```

---

## Development Phases

### Phase 1: Foundation (Week 1-2)
- [x] **T**: Port OT-2 control code into OT2Robot class
- [x] **T**: Port protocol execution + error recovery
- [x] **A**: Port vision detection pipeline
- [x] **A**: Implement GP surrogate model + acquisition functions
- [ ] **Y**: Define initial experiment parameters and target colors
- [ ] **D**: Generate synthetic color mixing dataset

### Phase 2: Core Pipeline (Week 3-4)
- [ ] **T**: Test robot commands against live OT-2
- [ ] **A**: Implement color extraction from well plate images
- [ ] **A**: Integrate vision into active learning pipeline
- [ ] **Y**: Finalize well plate layout and experiment protocol
- [ ] **D**: Build data preprocessing pipeline for real images

### Phase 3: Integration (Week 5-6)
- [ ] **T + A**: Complete end-to-end active learning loop
- [ ] **Y**: Run pilot experiments, document procedures
- [ ] **D**: Validate data pipeline with real experimental data

### Phase 4: Optimization & Report (Week 7-8)
- [ ] **All**: Run full active learning experiments
- [ ] **A**: Tune ML model, compare acquisition strategies
- [ ] **Y + D**: Analyze results, create visualizations
- [ ] **Y**: Write final report
- [ ] **All**: Code cleanup, documentation, final presentation
