# ACO-MMAS Based SVM Hyperparameter Optimization

This project applies **Ant Colony Optimization with Max-Min Ant System (ACO-MMAS)** to optimize a **Gaussian Kernel SVM** classifier on the **Ionosphere** dataset.

The goal is to find the best SVM hyperparameter combination for binary classification.

---

## Dataset

The project uses the **Ionosphere dataset**, which is available from the UCI Machine Learning Repository:

- Dataset link: [UCI Ionosphere Dataset](https://archive.ics.uci.edu/dataset/52/ionosphere)

In this MATLAB implementation, the built-in `ionosphere` dataset is used:

```matlab
load ionosphere
```

The dataset contains radar signal features and binary class labels:

```text
good
bad
```

Before training, zero-variance features are removed because they do not provide useful information for classification.

---

## Model

The classifier used in this project is:

```text
Gaussian Kernel Support Vector Machine
```

The optimized hyperparameters are:

| Hyperparameter | Description |
|---|---|
| `BoxConstraint` | Controls the trade-off between margin width and classification error. |
| `KernelScale` | Controls the spread of the Gaussian kernel. |
| `Standardize` | Determines whether the input features are standardized. |

---

## Methodology

The workflow consists of 11 main steps:

1. Load the Ionosphere dataset.
2. Split the data into training and test sets.
3. Create a 5-fold cross-validation partition.
4. Define the hyperparameter search space.
5. Set ACO-MMAS parameters.
6. Initialize the combined 3D pheromone and heuristic matrices.
7. Use Classification Learner output as the initial solution.
8. Run the ACO optimization loop.
9. Train and test the final SVM model.
10. Visualize convergence and confusion matrix.
11. Display the result table.

---

## ACO-MMAS Design

Unlike simple ACO implementations that use separate pheromone vectors for each hyperparameter, this project uses a **combined 3D pheromone matrix**:

```text
pheromone(BoxConstraint, KernelScale, Standardize)
```

This allows the algorithm to evaluate hyperparameter combinations together instead of independently.

The selection probability is based on both pheromone and heuristic information:

```text
probability = pheromone^alpha × heuristic^beta
```

MMAS limits are also used to prevent premature convergence:

```text
tau_min = 0.1
tau_max = 10.0
```

An early stopping mechanism is applied if no improvement is observed for 8 consecutive iterations.

---

## Hyperparameter Search Space

| Hyperparameter | Search Range |
|---|---:|
| `BoxConstraint` | 0.01 – 316.227766 |
| `KernelScale` | 0.1 – 31.622776 |
| `Standardize` | true / false |

---

## Experimental Setup

| Setting | Value |
|---|---:|
| Training set | 80% |
| Test set | 20% |
| Cross-validation | 5-fold |
| Number of ants | 20 |
| Maximum iterations | 30 |
| Early stopping patience | 8 |

---

## Results

The best hyperparameter combination found by ACO-MMAS was:

| Hyperparameter | Value |
|---|---:|
| `BoxConstraint` | 316.227766 |
| `KernelScale` | 3.651741 |
| `Standardize` | false |

Final performance:

| Metric | Result |
|---|---:|
| Training CV Accuracy | 95.73% |
| Test Accuracy | 91.43% |
| Completed Iterations | 17 / 30 |

The algorithm reached its best validation performance at iteration 5 and stopped at iteration 17 due to early stopping.

---

## Output

The script generates:

- ACO-SVM convergence plot
- Test confusion matrix
- Iteration-based result table

---

## How to Run

Open MATLAB and run:

```matlab
main_aco_svm_ionosphere_v2
```

---

## Requirements

- MATLAB
- Statistics and Machine Learning Toolbox

---

## References

- UCI Machine Learning Repository: [Ionosphere Dataset](https://archive.ics.uci.edu/dataset/52/ionosphere)
- MATLAB Documentation: `fitcsvm`
- MATLAB Documentation: `cvpartition`
- MATLAB Documentation: `crossval`
- MATLAB Documentation: `kfoldLoss`

---

## Summary

This project shows that **ACO-MMAS** can be used as an effective metaheuristic approach for optimizing SVM hyperparameters.

```text
Dataset       : Ionosphere
Model         : Gaussian Kernel SVM
Optimizer     : ACO-MMAS
CV Accuracy   : 95.73%
Test Accuracy : 91.43%
```