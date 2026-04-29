# Code Translation and Cross-Language Notes

## Overview

This project spans multiple languages and frameworks:
- **R**: Data generation (simulate_data.R)
- **Python**: ML models, RL, evaluation (main codebase)
- **Framework Variety**: scikit-learn, XGBoost, TensorFlow/PyTorch, ReAgent

This document tracks translation decisions and framework-specific implementation notes.

## 1. Data Generation Translation

### R Original: simulate_data.R

```r
library(dplyr)
library(caret)

n_buyer = 10
S = 375
A = 25
n_t ~ N(25, 5), min=10

# Output: 778 columns (id, states, actions, next_states, outcome)
```

### Python Translation Strategy

**Option A**: Use reticulate or rpy2 to call R directly
```python
import rpy2.robjects as robjects
robjects.r.source('mksc/3-Replication/5-Data/simulate_data.R')
```
**Pros**: Exact reproducibility
**Cons**: Requires R installation; slower

**Option B**: Translate R logic to Python (Recommended)
```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import label_binarize

def generate_data(n_buyer=10, S=375, A=25, seed=42):
    np.random.seed(seed)
    # ... translate R logic line-by-line
```
**Pros**: Pure Python; fast; self-contained
**Cons**: Potential numeric differences due to RNG differences

### Key Translation Points

#### 1.1 Random Number Generation

**R Code**:
```r
n_t = round(rnorm(n_buyer, 25, 5), 0)
state = matrix(rnorm(n_row*(S-1), 0, 1), nrow=n_row)
action = floor(runif(n_row, min=1, max=26))
```

**Python Equivalent**:
```python
n_t = np.round(np.random.normal(25, 5, n_buyer)).astype(int)
n_t = np.maximum(n_t, 10)  # pmax(n_t, 10)

state = np.random.normal(0, 1, size=(n_row, S-1))
state = np.round(state, 2)  # round to 2 decimals

action = np.floor(np.random.uniform(1, 26, n_row)).astype(int)
```

**⚠️ Numeric Differences**:
- R's `rnorm()` vs. Python's `np.random.normal()`: Different RNG algorithms
- Solution: Use same seed and accept small distributional differences
- Validation: Compare summary statistics, not exact values

#### 1.2 Data Frame Operations

**R Code**:
```r
df_1 <- data.frame(buyer_id)
df_1 <- df_1 %>% group_by(buyer_id) %>% mutate(receive_time_id = row_number())
dmy <- dummyVars(" ~ .", data = df_1)
df_2 <- data.frame(predict(dmy, newdata = df_1))
```

**Python Equivalent**:
```python
df = pd.DataFrame({'buyer_id': buyer_ids})
df['receive_time_id'] = df.groupby('buyer_id').cumcount() + 1

# One-hot encode actions
actions_onehot = pd.get_dummies(df['action'], prefix='a')
```

#### 1.3 Lead/Lag Operations

**R Code**:
```r
tmp2 <- tmp %>% mutate_at(.vars = 2:ncol(tmp), 
                        .funs = list(n = f_lead))
```

**Python Equivalent**:
```python
# Lead (shift -1) for next states
for col in state_cols:
    df[f'n{col}'] = df.groupby('buyer_id')[col].shift(-1)
```

### Data Generation Implementation Plan

**File**: `code/01_generate_data.py`

```python
def generate_simulated_data(n_buyer=10, S=375, A=25, seed=42):
    """
    Generate simulated data matching R's simulate_data.R
    
    Parameters:
    - n_buyer: Number of buyers
    - S: Number of state dimensions
    - A: Number of actions
    - seed: Random seed for reproducibility
    
    Returns:
    - df: DataFrame with 778 columns
    """
    np.random.seed(seed)
    
    # Step 1: Generate trajectory lengths
    n_t = np.maximum(np.round(np.random.normal(25, 5, n_buyer)).astype(int), 10)
    n_row = n_t.sum()
    
    # Step 2: Create base structure
    df = pd.DataFrame({
        'buyer_id': np.repeat(np.arange(1, n_buyer+1), n_t),
        'receive_time_id': np.concatenate([np.arange(1, t+1) for t in n_t])
    })
    
    # Step 3: Generate states (375 columns)
    state_df = pd.DataFrame(
        np.round(np.random.normal(0, 1, (n_row, S)), 2),
        columns=[f's{i+1}' for i in range(S)]
    )
    df = pd.concat([df, state_df], axis=1)
    
    # Step 4: Generate actions (25 columns, one-hot)
    actions = np.floor(np.random.uniform(1, 26, n_row)).astype(int)
    action_onehot = pd.get_dummies(actions, prefix='a', drop_first=False).astype(int)
    df = pd.concat([df, action_onehot], axis=1)
    
    # Step 5: Generate next states
    next_state_df = df[[f's{i+1}' for i in range(S)]].copy()
    next_state_df = next_state_df.reset_index(drop=True).shift(-1)
    next_state_df.columns = [f'ns{i+1}' for i in range(S)]
    df = pd.concat([df, next_state_df], axis=1)
    
    # Step 6: Generate outcomes
    df['div_pay_amt_fillna'] = np.maximum(0, np.random.normal(-2, 2, n_row))
    
    return df
```

### Validation Against Original Data

**File**: `code/01_generate_data.py` includes validation function

```python
def validate_simulated_data(df, expected_rows=None):
    """
    Validate generated data structure
    """
    errors = []
    
    # Check shape
    if df.shape[1] != 778:
        errors.append(f"Column count: expected 778, got {df.shape[1]}")
    
    # Check columns
    expected_cols = ['buyer_id', 'receive_time_id']
    expected_cols += [f's{i+1}' for i in range(375)]
    expected_cols += [f'a{i+1}' for i in range(25)]
    expected_cols += [f'ns{i+1}' for i in range(375)]
    expected_cols += ['div_pay_amt_fillna']
    
    missing_cols = set(expected_cols) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")
    
    # Check data types
    if df['buyer_id'].dtype not in [np.int64, np.int32, int]:
        errors.append(f"buyer_id dtype: expected int, got {df['buyer_id'].dtype}")
    
    # Check value ranges
    if (df['div_pay_amt_fillna'] < 0).any():
        errors.append("Outcome has negative values (should be >= 0)")
    
    return errors if errors else "✓ Validation passed"
```

## 2. Supervised Learning Models Translation

### 2.1 Linear Model

**Framework**: scikit-learn vs. statsmodels

```python
# Option 1: scikit-learn (prediction focus)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Option 2: statsmodels (inference focus)
import statsmodels.api as sm
X_train = sm.add_constant(X_train)
model = sm.OLS(y_train, X_train).fit()
print(model.summary())
```

**Choice**: Use both:
- scikit-learn for prediction metrics
- statsmodels for coefficient inference

### 2.2 GBDT Model

**Framework Options**:

```python
# XGBoost
import xgboost as xgb
model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)

# LightGBM
import lightgbm as lgb
model = lgb.LGBMRegressor(
    n_estimators=100,
    num_leaves=31,
    learning_rate=0.1,
    random_state=42
)
```

**Decision**: Use XGBoost (more standard in literature)

**Key Parameters**:
- `n_estimators=100`: Number of trees
- `max_depth=6`: Tree depth
- `learning_rate=0.1`: Shrinkage parameter

### 2.3 Deep Neural Network

**Framework**: TensorFlow/Keras

```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(n_features,)),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # Output layer for regression
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=100, batch_size=32, 
         validation_split=0.1, verbose=1)
```

**Architecture Choices**:
- Input: n_state_features + n_action_features
- Hidden: [128, 64, 32] neurons
- Activation: ReLU
- Regularization: BatchNorm + Dropout
- Output: Single neuron (regression)

### 2.4 Orthogonal Random Forest (ORF)

**Challenge**: Limited ORF implementations in Python

**Solution**: Use double machine learning approach

```python
from sklearn.ensemble import RandomForestRegressor

def orthogonal_random_forest(X, y, T, cv_folds=5):
    """
    Simplified ORF implementation
    X: features, y: outcome, T: treatment
    """
    # Estimate nuisance parameters on fold 1
    # ... cross-fit approach
    pass
```

**Alternative**: Skip ORF if resources limited; focus on Linear, GBDT, DNN

## 3. Reinforcement Learning Implementation

### 3.1 BCQ Implementation

**Framework**: PyTorch

**Key Components**:

1. **Q-Network**: 
   - Input: state features
   - Output: Q-value for each action
   - Architecture: 2-3 hidden layers, ReLU activation

2. **Policy Network**:
   - Input: state
   - Output: action probabilities
   - Constraint: KL divergence from behavior policy

3. **Experience Replay**:
   - Store (s, a, r, s', done) tuples
   - Sample minibatches for training

**Implementation Outline**:

```python
import torch
import torch.nn as nn
import torch.optim as optim

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, state):
        return self.net(state)

class BCQAgent:
    def __init__(self, state_dim, action_dim, gamma=0.95):
        self.q_network = QNetwork(state_dim, action_dim)
        self.gamma = gamma
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
    
    def update(self, batch):
        s, a, r, s_next, done = batch
        # Compute target: r + γ * max_a Q(s_next, a)
        target = r + self.gamma * self.q_network(s_next).max(dim=1)[0] * (1 - done)
        
        # Compute loss: (Q_pred - target)^2
        q_pred = self.q_network(s).gather(1, a.unsqueeze(1))
        loss = nn.functional.mse_loss(q_pred.squeeze(), target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
```

### 3.2 Alternative: Use ReAgent

**If Available**:
```python
from reagent.agents.qrdqn_agent import QRDQNAgent
from reagent.core import Parameter

agent = QRDQNAgent(...)
agent.train(data_loader)
```

**If Not Available**:
- Use custom PyTorch implementation
- Document limitations
- Cross-reference with original implementation

## 4. Off-Policy Evaluation: Doubly Robust

### 4.1 Importance Sampling

```python
def importance_sampling(trajectories, behavior_policy, eval_policy, gamma=0.95):
    """
    Compute IS estimate of policy value
    """
    values = []
    
    for traj in trajectories:
        weight = 1.0
        value = 0.0
        
        for t, (s, a, r, s_next) in enumerate(traj):
            # Importance weight
            pi_e_a = eval_policy(s)[a]
            pi_b_a = behavior_policy(s)[a]
            weight *= (pi_e_a / pi_b_a)
            
            # Accumulate discounted reward
            value += (gamma ** t) * weight * r
        
        values.append(value)
    
    return np.mean(values), np.std(values)
```

### 4.2 Doubly Robust Estimator

```python
def doubly_robust_estimate(trajectories, q_model, v_model, 
                          behavior_policy, eval_policy, gamma=0.95):
    """
    Compute DR estimate combining model and weighting
    """
    values = []
    
    for traj in trajectories:
        weight = 1.0
        value = 0.0
        
        for t, (s, a, r, s_next) in enumerate(traj):
            # Model-based term
            q_value = q_model(s, a)
            
            # Next state value
            v_next = v_model(s_next)
            
            # Importance weight
            pi_e_a = eval_policy(s)[a]
            pi_b_a = behavior_policy(s)[a]
            weight *= (pi_e_a / pi_b_a)
            
            # DR combines both
            dr_term = weight * (r - q_value) + v_next
            value += (gamma ** t) * dr_term
        
        values.append(value)
    
    return np.mean(values), np.std(values)
```

### 4.3 Variance Reduction Techniques

**Weighted IS** (WPDIS):
```python
def weighted_is(trajectories, ...):
    # Normalize weights
    weights_normalized = weights / np.sum(weights)
    value = np.mean(weights_normalized * rewards)
    return value
```

## 5. Version Management and Compatibility

### Framework Version Matrix

| Framework | Version | Python | Compatibility |
|-----------|---------|--------|----------------|
| numpy | 1.21.6 | 3.8+ | Legacy support |
| pandas | 1.3.5 | 3.8+ | Standard |
| scikit-learn | 1.0.2 | 3.8+ | Standard |
| xgboost | 1.5.2 | 3.8+ | GPU support optional |
| tensorflow | 2.10.0 | 3.9-3.10 | Latest stable |
| torch | 1.12.1 | 3.8-3.10 | CPU or CUDA |

### Potential Conflicts

**Issue 1**: TensorFlow vs PyTorch CUDA
- **Solution**: Use separate environments or CPU-only

**Issue 2**: scikit-learn deprecations
- **Solution**: Pin version 1.0.2; replace deprecated imports

**Issue 3**: pandas API changes
- **Solution**: Pin version 1.3.5; use backward-compatible syntax

## 6. Known Issues and Workarounds

### Issue 1: Random Seed Differences

**Problem**: R and Python RNGs produce different sequences even with same seed

**Workaround**:
```python
# Seed everything
np.random.seed(42)
tf.random.set_seed(42)
torch.manual_seed(42)
```

**Validation**: Compare summary statistics, not exact values

### Issue 2: XGBoost Deprecated Imports

**Problem**: sklearn.externals removed in recent versions

**Original**: `from sklearn.externals import joblib`

**Fix**: `import joblib` (separate package)

### Issue 3: Batch RL Convergence

**Problem**: RL training may not converge on synthetic data

**Mitigation**:
- Use curriculum learning (gradual difficulty increase)
- Adjust hyperparameters (learning rate, γ)
- Monitor training loss plots
- Document non-convergence as limitation

## 7. Documentation Checklist

- [ ] All random seeds documented
- [ ] Framework versions pinned
- [ ] RNG differences acknowledged
- [ ] Translation decisions justified
- [ ] Known issues listed with workarounds
- [ ] Cross-language interfaces specified
- [ ] Test cases for each module

## 8. Reference Implementations

**BCQ Paper**: Fujimoto et al. (2019)
- GitHub: https://github.com/sfujim/BCQ
- Reference implementation in PyTorch

**Doubly Robust**: Dudik et al. (2014)
- Multiple implementations available
- Ensure mathematical correctness

**Off-Policy Evaluation Library**: Voigt et al. (OPE)
- Python library for common OPE estimators
- May be useful for validation
