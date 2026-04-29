# Step-by-Step Replication Instructions

## Prerequisites

- Python 3.8+
- R (for optional cross-validation of data generation)
- pip package manager

## Phase 1: Environment Setup

### Step 1.1 - Install Dependencies

```bash
cd replication_project
pip install -r requirements.txt
```

This will install:
- Data processing: pandas, numpy
- ML models: scikit-learn, xgboost, lightgbm
- Deep learning: tensorflow or pytorch
- Statistical tools: statsmodels
- Visualization: matplotlib, seaborn
- RL tools: ReAgent (if available)

### Step 1.2 - Verify Installation

```bash
python -c "import pandas; import numpy; import sklearn; print('Dependencies OK')"
```

## Phase 2: Data Generation and Validation

### Step 2.1 - Generate Simulated Data

Run the data generation script:
```bash
python code/01_generate_data.py
```

This will:
- Load or execute `simulate_data.R` (or Python equivalent)
- Generate simulated user interaction data
- Save to `data/raw/simulated_data.txt`
- Generate summary statistics in `logs/data_generation.log`

**CHECKPOINT 1: Data Generation Review**
- [ ] Check `logs/data_generation.log` for errors
- [ ] Verify `data/raw/simulated_data.txt` has correct structure
- [ ] Review summary statistics in output
- [ ] Confirm N observations and variable counts

### Step 2.2 - Data Validation

The script will verify:
- Number of observations: Should be ~n_buyer × avg_n_t
- Columns: buyer_id, receive_time_id, state variables (s1-s375), action (a1-a25), next_state variables (ns1-ns375), outcome (div_pay_amt_fillna)
- Data types: Numeric for states, integer for actions
- Missing values: Handled appropriately

If validation fails, check `notes/materials_review.md` for troubleshooting.

## Phase 3: Model-Free Evidence

### Step 3.1 - Run Model-Free Analysis

```bash
python code/02_model_free.py
```

This generates:
- Descriptive statistics tables
- Visualizations (Figures 9, 10, 12, 13, etc.)
- Output files in `output/figures/`
- Detailed log in `logs/model_free.log`

**CHECKPOINT 2: Model-Free Review**
- [ ] Compare figures against original paper
- [ ] Check action distribution statistics
- [ ] Review outcome variable distributions
- [ ] Verify consistency with reported findings

## Phase 4: Model Replication

### Step 4.1 - Linear Model

```bash
python code/03_ml_models.py --model linear
```

Outputs:
- Coefficients and significance tests
- Prediction metrics (RMSE, R²)
- Results to `output/tables/linear_results.csv`

### Step 4.2 - GBDT Model

```bash
python code/03_ml_models.py --model gbdt
```

Outputs:
- Feature importance rankings
- Prediction metrics
- Results to `output/tables/gbdt_results.csv`

### Step 4.3 - DNN Model

```bash
python code/03_ml_models.py --model dnn
```

Outputs:
- Network architecture and training history
- Prediction metrics
- Results to `output/tables/dnn_results.csv`

## Phase 5: Advanced Models

### Step 5.1 - ORF (Orthogonal Random Forest)

```bash
python code/03_ml_models.py --model orf
```

## Phase 6: Reinforcement Learning & Policy Evaluation

### Step 6.1 - Doubly Robust Estimation

```bash
python code/04_doubly_robust.py
```

This is the most computationally intensive step. Outputs:
- V values (state values)
- Q values (action values)
- DR estimates (doubly robust policy value)
- Importance sampling estimates
- Results to `output/tables/policy_evaluation.csv`

### Step 6.2 - RL-Based Models (BCQ)

```bash
python code/05_rl_models.py
```

Outputs:
- Trained policy network
- Policy value estimates
- Model comparison results

**CHECKPOINT 3: Policy Evaluation Review**
- [ ] Compare DR estimates against original paper
- [ ] Review policy value differences (expected: <10% deviation)
- [ ] Check convergence diagnostics in logs
- [ ] Validate IS/IPS/DR estimate relationships

## Phase 7: Results Compilation

### Step 7.1 - Generate Comparison Report

All comparison tables are compiled into:
- `output/tables/replication_comparison.csv`
- `notes/replication_comparison.md` (markdown summary)

### Step 7.2 - Generate Final Report

```bash
python code/generate_report.py
```

This creates:
- `report/REPLICATION_REPORT.md`
- `report/REPLICATION_REPORT.html`

## Troubleshooting

### Data Generation Issues
- Check `logs/data_generation.log`
- Verify R/Python environment
- See `notes/materials_review.md`

### Model Fitting Issues
- Check `logs/` directory for detailed error traces
- Verify data preprocessing in `code/03_ml_models.py`
- See `notes/code_translation.md` for known issues

### RL Model Issues
- These are computationally intensive; allow extra time
- Check GPU availability and CUDA setup
- See `notes/code_translation.md` for framework-specific notes

## Expected Runtime

- Data generation: ~1-5 minutes
- Model-free analysis: ~2-5 minutes
- ML models: ~10-30 minutes
- DR estimation: ~30-60 minutes (can be parallelized)
- RL models: ~60-120+ minutes (depends on hardware)
- **Total: 2-4 hours**

## Output Validation

After completion, verify:
1. All output files exist in `output/`
2. No error messages in logs
3. Results match expected value ranges (see paper)
4. All figures are generated and readable

## Next Steps

1. Review `notes/replication_comparison.md` for detailed result comparison
2. Check `report/` for final summary
3. Document any deviations in `notes/` as needed
4. Create GitHub issue if significant discrepancies found
