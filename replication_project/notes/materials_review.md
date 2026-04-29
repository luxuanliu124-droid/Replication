# Materials and Assumptions Review

## Available Materials from mksc/3-Replication

### 1. Data Generation
- **Location**: `mksc/3-Replication/5-Data/`
- **Files**:
  - `simulate_data.R`: R script to generate synthetic data
  - `simulated_data.txt`: Pre-generated simulated dataset (CSV format)
- **Status**: ✓ Available

### 2. Model-Free Evidence
- **Location**: `mksc/3-Replication/1-Model Free Evidence/`
- **Files**:
  - `Figure_9_10.R`: R code for generating Figures 9 & 10
  - `Figure_A5_A6.R`: R code for appendix figures
  - `Figure_Submit_08082022.ipynb`: Jupyter notebook with visualizations
  - `Gamma_tau.csv`: Analysis data
  - `figure9.csv`: Large data file for Figure 9
  - Excel files: Figure outputs (xlsx format)
  - `Readme_FigureTableSource_08282022.xlsx`: Documentation
- **Status**: ✓ Available

### 3. Supervised Learning Models
- **Location**: `mksc/3-Replication/2-Model/`
- **Structure**:
  - `1-Linear/`: Linear model code
  - `2-GBDT/`: Gradient boosted tree code
  - `3-DNN/`: Deep neural network code
  - `4-ORF/`: Orthogonal random forest code
  - `6-BDRL/`: Batch deep RL code
  - `input_file.py`: Input data path configuration
  - Chinese documentation: `直播带货优惠券代码Readme_031520.docx`
- **Status**: ⚠ Subdirectories exist but may need file listing

### 4. Doubly Robust Evaluation
- **Location**: `mksc/3-Replication/3-Doubly Robust/`
- **Files**:
  - `main_live_working_log.py`: Main evaluation script (verified)
  - `live_domain/`: Supporting modules
  - `src/`: Utility modules
- **Status**: ✓ Available

### 5. Documentation
- **Location**: `mksc/3-Replication/`
- **Files**:
  - `Code Readme.docx`: Main documentation (English)
  - Chinese docs in 2-Model subdirectory
- **Status**: ⚠ Needs review

## Data Structure Assumptions

### Based on simulate_data.R Analysis

**Total Rows**: n_row = sum(n_t)
- n_buyer = 10 (or 1,000,000 in production)
- n_t ~ N(25, 5), min=10
- Expected: n_row ≈ 250 (for n_buyer=10)

**Column Structure**:

1. **Identifier Columns** (2):
   - `buyer_id`: 1, 2, ..., n_buyer
   - `receive_time_id`: Row number within each buyer

2. **State Columns** (375 = S):
   - s1, s2, ..., s374: User features ~ N(0, 1)
   - s375: Churn indicator (binary)

3. **Action Column** (25 = A, one-hot encoded):
   - a1, a2, ..., a25: One-hot encoding of coupon choice
   - Action generated as: floor(runif(n_row, 1, 26)) → one-hot encoded

4. **Next State Columns** (375):
   - ns1, ns2, ..., ns375: Lead state (next observation)
   - Last row per buyer: NA (end of trajectory)

5. **Outcome Column** (1):
   - `div_pay_amt_fillna`: max(0, N(-2, 2))
   - Represents purchase value or indicator

**Total Expected Columns**: 2 + 375 + 25 + 375 + 1 = 778

### Data Validation Checks

```python
# Expected structure after loading
assert df.shape[1] == 778, f"Expected 778 columns, got {df.shape[1]}"
assert 'buyer_id' in df.columns
assert 'receive_time_id' in df.columns
assert all(f's{i}' in df.columns for i in range(1, 376)), "Missing state columns"
assert all(f'a{i}' in df.columns for i in range(1, 26)), "Missing action columns"
assert all(f'ns{i}' in df.columns for i in range(1, 376)), "Missing next-state columns"
assert 'div_pay_amt_fillna' in df.columns, "Missing outcome column"

# Data type checks
assert df['buyer_id'].dtype in [int, 'int64', 'int32']
assert df['receive_time_id'].dtype in [int, 'int64', 'int32']
assert all(df[[f's{i}' for i in range(1, 376)]].dtypes == float), "State columns should be float"
assert all(df[[f'a{i}' for i in range(1, 26)]].dtypes in [int, float]), "Action columns should be numeric"

# Logical checks
assert (df['div_pay_amt_fillna'] >= 0).all(), "Outcome should be non-negative"
assert df['buyer_id'].min() >= 1
assert df['receive_time_id'].min() >= 1
```

## Key Assumptions Made

### 1. Data Generation

**Assumption 1.1**: Random seed reproducibility
- The original R code generates deterministic output with fixed seed
- Python translation must use same seed strategy
- **Action**: Document seed values in code

**Assumption 1.2**: Normalization
- State variables are normalized ~ N(0, 1)
- No additional scaling is applied in data generation
- Models may require standardization during training

**Assumption 1.3**: Missing values
- Last observation per buyer has NA in next-state columns
- Outcome is always defined (no NA)
- **Action**: Handle NA gracefully in state columns

### 2. Model Specifications

**Assumption 2.1**: Outcome Model
- Outcome depends on (state, action) but also previous actions (Markovian with memory)
- State transitions are independent of action (state evolution doesn't depend on coupon)
- **Challenge**: Violates strict Markov assumption; may need longer state history

**Assumption 2.2**: Behavior Policy
- Historical action distribution is the behavior policy
- Estimated from frequency of actions in data
- Used for importance weighting

**Assumption 2.3**: Action Space
- Actions are discrete (25 coupons)
- All actions are feasible for all states
- No constraints on action availability

### 3. ML Model Assumptions

**Assumption 3.1**: Linear Model
- Linear relationship between features and outcome
- Estimated via OLS regression
- Baseline for comparison

**Assumption 3.2**: GBDT Model
- XGBoost or LightGBM implementation
- Default hyperparameters used (hyperparameter tuning TBD)
- Can capture non-linear relationships

**Assumption 3.3**: DNN Model
- Standard feedforward architecture
- ReLU activations, batch normalization
- Trained with Adam optimizer
- Convergence checked via dev set

### 4. RL/Evaluation Assumptions

**Assumption 4.1**: Discount Factor
- γ = 0.95 (or as specified in config)
- Standard choice for finite horizon tasks

**Assumption 4.2**: Off-policy Correctness
- POMDP with full observability of state
- No unobserved confounders
- Validity of IPW requires positivity (π_b(a|s) > 0 for all observed (s,a))

**Assumption 4.3**: Doubly Robust Estimation
- Both model and weighting are used
- Provides robustness if either is well-specified
- Assumes finite sample corrections for variance

## Documentation of Deviations

### Deviation 1: Data Format
- **Original**: R data.frame, written to CSV with pandas one-hot encoding
- **Current**: Python loads CSV directly
- **Impact**: Minimal if CSV format preserved

### Deviation 2: Random Seed
- **Original**: R's set.seed() in simulate_data.R
- **Current**: Python's np.random.seed() or scipy.stats
- **Impact**: Same distributions but different sequence of random numbers
- **Mitigation**: Document seed values; results should match in distribution

### Deviation 3: Deep Learning Framework
- **Original**: Possibly TensorFlow 1.x or Keras
- **Current**: TensorFlow 2.10.0 or PyTorch
- **Impact**: Potential numerical differences in weight initialization and optimization
- **Mitigation**: Match architecture exactly; seed all RNGs

### Deviation 4: RL Framework
- **Original**: Likely custom implementation or ReAgent
- **Current**: Custom PyTorch implementation or ReAgent if available
- **Impact**: Algorithm differences if implementations vary
- **Mitigation**: Document algorithm carefully; validate on simple cases

## Package Version Pinning

### Rationale
Exact reproducibility requires fixed package versions. See `requirements.txt` for pinned versions.

### Critical Packages

| Package | Version | Reason |
|---------|---------|--------|
| pandas | 1.3.5 | Data I/O compatibility |
| numpy | 1.21.6 | Numerical consistency |
| scikit-learn | 1.0.2 | ML model algorithms |
| xgboost | 1.5.2 | GBDT implementation |
| tensorflow | 2.10.0 | DNN training |
| torch | 1.12.1 | Alternative DL framework |

### Dependency Conflicts

⚠️ **Potential Issues**:
- TensorFlow and PyTorch may have conflicting CUDA requirements
- XGBoost may require specific C++ library versions
- ReAgent has complex dependency tree

**Resolution**: Use virtual environment (`venv` or `conda`)

## Missing or Unclear Materials

### To-Do: Explore Subdirectories

- [ ] `mksc/3-Replication/2-Model/1-Linear/`: Find actual Linear model code
- [ ] `mksc/3-Replication/2-Model/2-GBDT/`: Find GBDT implementation
- [ ] `mksc/3-Replication/2-Model/3-DNN/`: Find DNN implementation
- [ ] `mksc/3-Replication/2-Model/4-ORF/`: Find ORF code
- [ ] `mksc/3-Replication/2-Model/6-BDRL/`: Find RL implementation
- [ ] `mksc/3-Replication/3-Doubly Robust/live_domain/`: Explore supporting modules
- [ ] `mksc/3-Replication/3-Doubly Robust/src/`: Find utility modules

## Environment Setup Summary

**Status**: ⚠️ Incomplete - need to explore subdirectories

**Next Steps**:
1. List files in each model subdirectory
2. Identify actual code files vs. output files
3. Map out dependencies between modules
4. Create consolidated requirements.txt
5. Document any missing components

**Critical Path**:
1. Verify data generation works independently
2. Map model-free analysis pipeline
3. Connect supervised learning models
4. Integrate RL/evaluation pipeline
5. Create unified execution script
