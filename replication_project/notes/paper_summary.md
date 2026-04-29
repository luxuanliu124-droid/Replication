# Research Paper Summary

## Main Research Question

How can reinforcement learning (specifically Batch Constrained Q-learning) improve coupon targeting policies in live-streaming commerce compared to traditional supervised learning approaches, and how can off-policy evaluation techniques (doubly robust estimation) validate the performance of learned policies?

## Problem Importance: Coupon Targeting in Marketing

### Why Coupon Targeting Matters:

1. **Business Context**: In live-streaming commerce, sellers need to decide when and which coupon to offer to maximize customer engagement and purchase value
2. **Optimization Challenge**: Different customers have different sensitivities; a one-size-fits-all policy is suboptimal
3. **Learning Constraints**: Historical data reflects past coupon policies, not optimal policies (off-policy problem)
4. **Scale**: With millions of transactions, even small improvements compound to significant business impact

## Key Methodological Components

### 1. Model-Free Evidence
- **Purpose**: Establish baseline understanding through descriptive statistics
- **Methods**: Summary statistics, conditional expectations, graphical analysis
- **Output**: Figures showing coupon usage patterns, outcome distributions

### 2. Supervised Machine Learning
- **Purpose**: Learn conditional outcome models from historical data
- **Models**:
  - **Linear**: Simple, interpretable baseline
  - **GBDT**: Gradient Boosted Decision Trees (nonlinear, feature interactions)
  - **DNN**: Deep Neural Networks (flexible, high capacity)
  - **ORF**: Orthogonal Random Forest (double machine learning)

- **Use Case**: Predict likely outcomes given state and action (basis for other methods)

### 3. Batch Reinforcement Learning (BCQ)
- **Problem**: Standard RL assumes interaction with environment; here we only have logged data
- **Solution**: Batch Constrained Q-learning ensures learned policy doesn't deviate too far from behavior policy
- **Key Idea**: Avoid overestimating Q-values for out-of-distribution state-action pairs
- **Role in Study**: Learn an improved coupon targeting policy from historical data

### 4. Off-Policy Evaluation (OPE)
- **Problem**: Can't A/B test new policies on all data; need to estimate policy value from logs
- **Solution**: Doubly Robust (DR) estimator combines:
  - **Direct Method**: Model-based value estimation
  - **Inverse Probability Weighting (IPW)**: Importance sampling correction
  - **Robustness**: Works if either method is correct, not both required

## Data Structure: Simulated User-Level Interaction Data

### Data Generation Process

The project uses simulated data (see `simulate_data.R`):

```
n_buyer = 10 (or 1,000,000 in full study)
S = 375 (state dimensions)
A = 25 (action dimensions: coupon types)
n_t ~ N(25, 5) (observations per buyer, min 10)
```

### Key Variables

1. **Identifiers**:
   - `buyer_id`: Unique buyer identifier
   - `receive_time_id`: Time index for each buyer

2. **State Variables** (s1, s2, ..., s375):
   - User characteristics (purchase history, browsing, engagement, demographics)
   - Last state dimension typically is churn indicator
   - Simulated as: `N(0, 1)` random normal variables

3. **Action** (a1, a25):
   - Coupon assignment (25 possible coupons)
   - One-hot encoded in implementation

4. **Outcome** (`div_pay_amt_fillna`):
   - Purchase value or purchase indicator
   - Simulated as: `max(0, N(-2, 2))` (right-censored at 0)

5. **Next State Variables** (ns1, ns375):
   - State after interaction (lead state)
   - Used for computing state transitions

### Data Structure Example

```
Columns: buyer_id | receive_time_id | s1 | ... | s375 | action | ns1 | ... | ns375 | div_pay_amt_fillna
Row 1:   1        | 1               | 0.5| ... | -0.2 | 5      | 0.3 | ... | -0.1  | 2.5
Row 2:   1        | 2               | 0.3| ... | -0.1 | 3      | 0.1 | ... | 0.2   | 1.2
...
```

## Modeling Pipeline

### Phase 1: Prediction Models

Train models to predict outcome given state and action:

```python
Outcome = f(state, action) + noise
```

**Models Implemented**:
- Linear regression
- GBDT (XGBoost, LightGBM)
- DNN (TensorFlow/PyTorch)

**Output**: Outcome predictions for all state-action pairs

### Phase 2: Policy Learning

Use RL to learn improved policy:

**Batch RL Approach**:
1. Learn Q-function: Q(s, a) = expected outcome for state s, action a
2. Extract policy: π(a|s) = argmax_a Q(s, a) (or softmax variant)
3. Constraint: Keep policy close to behavior policy to avoid extrapolation errors

**BCQ Implementation**:
- Learn both Q-function and policy
- Add regularization to limit deviation from behavior policy
- Typically uses twin networks for stability

### Phase 3: Policy Evaluation

Estimate value of learned policy using doubly robust estimator:

**Importance Sampling (IPS)**:
```
V_IPS = (1/N) Σ_trajectories [ w_t * reward_t ]
where w_t = π_e(a|s) / π_b(a|s) (importance weight)
```

**Model-Based (Direct)**:
```
V_Direct = (1/N) Σ [Q_model(s, π_e(s))]
```

**Doubly Robust**:
```
V_DR = (1/N) Σ [ w_t * (r_t - Q(s,a)) + Q_model(s, π_e(s)) ]
```

Benefits from both high-quality prediction and importance weighting.

## Core Findings (Expected)

### Main Results

1. **RL Outperforms Baselines**: BCQ-learned policy shows higher estimated value than:
   - Random policy (baseline)
   - Behavior policy (historical)
   - Simple supervised learning policies

2. **Doubly Robust Validation**: DR estimator provides:
   - More stable estimates than IPS alone
   - Better control over bias-variance tradeoff
   - Confidence in policy value estimates

3. **Model Comparison**:
   - GBDT and DNN typically outperform Linear
   - Feature interactions are important
   - RL component captures non-linear policy effects

## Key References (To Be Verified)

- Batch Constrained Q-learning (Fujimoto et al., 2019)
- Off-policy evaluation: Doubly Robust Estimators (Dudik et al., 2014)
- Reinforcement Learning in Marketing (Misra et al., 2018)
- Causal forests / Orthogonal Random Forests (Athey & Wager, 2019)

## Implementation Notes

### Cross-Language Considerations

- **R Components**: Data generation (simulate_data.R)
- **Python Components**: ML models, RL, evaluation (main code)
- **Challenge**: Data format consistency across languages
- **Solution**: Use CSV/standard formats as interchange (simulated_data.txt)

### Known Limitations

1. **Simulated Data**: Synthetic data may not capture real market dynamics
2. **Computational**: RL training is computationally intensive
3. **Extrapolation**: Policy learning limited by coverage of behavior policy
4. **Framework Availability**: Some RL frameworks (ReAgent) may not be readily available

## To-Do: Verification Tasks

- [ ] Locate all papers listed in references
- [ ] Verify paper details (author, year, journal)
- [ ] Mark citations as "Verified" once confirmed
- [ ] Document any unverifiable references
