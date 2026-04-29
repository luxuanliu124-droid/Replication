# Coupon Targeting Replication Project

This project replicates research on reinforcement learning-based coupon targeting in live-streaming commerce settings.

## Project Overview

The replication focuses on:
1. **Model-Free Evidence** - Descriptive statistics and preliminary analysis
2. **Supervised Learning Models** - Linear, GBDT, DNN models
3. **Advanced Models** - ORF, structural models
4. **Reinforcement Learning** - BCQ (Batch Constrained Q-learning) implementation
5. **Policy Evaluation** - Doubly robust estimation of policy value

## Directory Structure

```
replication_project/
  README.md                    # This file
  INSTRUCTIONS.md              # Step-by-step replication guide
  requirements.txt             # Python dependencies
  original/                    # Original materials (unmodified)
    readme/
    code/
    data/
  data/                        # Data directory
    raw/                       # Raw simulated data
    processed/                 # Processed/cleaned data
  code/                        # Analysis code
    01_generate_data.py        # Data generation
    02_model_free.py           # Model-free evidence
    03_ml_models.py            # Supervised learning models
    04_doubly_robust.py        # Doubly robust estimation
    05_rl_models.py            # RL-based models (BCQ)
    utils.py                   # Utility functions
  notes/                       # Documentation
    paper_summary.md           # Research paper summary
    materials_review.md        # Materials and assumptions
    code_translation.md        # Cross-language translation notes
    replication_comparison.md  # Results comparison
  output/                      # Output files
    tables/                    # Result tables
    figures/                   # Generated figures
  logs/                        # Execution logs
  report/                      # Final replication report
```

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate simulated data:**
   ```bash
   python code/01_generate_data.py
   ```

3. **Run analysis pipeline:**
   ```bash
   python code/02_model_free.py
   python code/03_ml_models.py
   python code/04_doubly_robust.py
   python code/05_rl_models.py
   ```

4. **Check results:**
   Results are saved in `output/` directory with detailed logs in `logs/`

## Key Files

- **INSTRUCTIONS.md** - Detailed step-by-step instructions
- **notes/paper_summary.md** - Understanding of the research
- **notes/materials_review.md** - Materials and data structure
- **notes/code_translation.md** - Python translation notes
- **notes/replication_comparison.md** - Results vs original

## Reproducibility

This project is designed for full reproducibility:
- Fixed random seeds
- All dependencies pinned to specific versions
- Detailed documentation of all assumptions
- Checkpoint system for human review

## Contact & Support

For questions about this replication, refer to the detailed notes in the `notes/` directory.
