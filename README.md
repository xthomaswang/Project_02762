# 02762_A_Project

**Automated Colorimetric Assay Optimization via Active Learning**

## Overview

This project uses the OT-2 liquid handling robot to automatically mix red, green, and blue dyes in a 96-well plate, captures images with a camera for color feedback, and applies active learning (ML) to iteratively optimize dye formulations toward target colors — simulating a colorimetric assay workflow.

## Project Structure

```
02762_A_Project/
├── robot/                  # OT-2 robot control
│   ├── protocols/          # Opentrons protocol scripts
│   └── configs/            # Robot and labware configurations
├── vision/                 # Computer vision pipeline
│   ├── preprocessing/      # Image preprocessing (crop, normalize)
│   ├── detection/          # Well plate and well detection
│   └── color_extraction/   # RGB/HSV color extraction from wells
├── ml/                     # Machine learning / Active learning
│   ├── models/             # Surrogate models (GP, NN, etc.)
│   ├── acquisition/        # Acquisition functions (EI, UCB, etc.)
│   └── utils/              # ML utilities and helpers
├── integration/            # System integration (end-to-end pipeline)
├── data/                   # Data storage
│   ├── raw/                # Raw images and robot logs
│   ├── processed/          # Extracted color data
│   └── synthetic/          # Synthetic/simulated data for testing
├── experiments/            # Experiment management
│   ├── configs/            # Experiment configuration files
│   └── results/            # Experiment results and logs
├── docs/                   # Documentation and reports
├── tests/                  # Unit and integration tests
│   ├── test_robot/
│   ├── test_vision/
│   ├── test_ml/
│   └── test_integration/
├── scripts/                # Utility scripts (setup, data gen, etc.)
└── notebooks/              # Jupyter notebooks for exploration
```

## Setup

```bash
# Clone the repository
git clone https://github.com/xthomaswang/02762_A_Project.git
cd 02762_A_Project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
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
