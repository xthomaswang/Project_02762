# Project_02762

**Automated Colorimetric Assay Optimization via Active Learning**

## Overview

This project uses the OT-2 liquid handling robot to automatically mix red, green, and blue dyes in a 96-well plate, captures images with a camera for color feedback, and applies active learning (ML) to iteratively optimize dye formulations toward target colors — simulating a colorimetric assay workflow.

## Project Structure

```
Project_02762/
├── src/                     # Project-specific source code
│   ├── ml.py                # GP surrogate model + acquisition functions
│   └── pipeline.py          # End-to-end active learning loop
├── configs/                 # Experiment configuration files
│   └── experiment.yaml
├── tests/                   # Unit and integration tests
├── data/                    # Runtime data (images, logs)
├── scripts/                 # CLI entry points
│   └── run_experiment.py
├── notebooks/               # Jupyter notebooks for exploration
└── docs/                    # Documentation
    └── TASK_DISTRIBUTION.md
```

Robot control, protocol execution, vision, and recovery are provided by the
[OpenOT2](https://github.com/xthomaswang/OpenOT2) package (installed as a dependency).

## Setup

```bash
# Clone the repository
git clone https://github.com/xthomaswang/Project_02762.git
cd Project_02762

# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run an active learning experiment
python scripts/run_experiment.py --config configs/experiment.yaml

# Or import modules directly
python -c "from openot2 import OT2Client; print('OK')"
```

## Team

| Member | Role |
|--------|------|
| **T** | System Integration + Robot Control |
| **A** | Computer Vision + ML Pipeline |
| **Y** | Experiment Design + Documentation |
| **D** | Data Processing + Experiment Support |

## License

This project is for academic use (CMU 02-762 Lab Methods, Spring 2026).
