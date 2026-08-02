# Bank marketing: predicting term-deposit subscription

Binary classification on the Kaggle Playground bank marketing dataset
(750,000 rows): given a customer's demographics, financial position and
contact history, predict whether they subscribe to a term deposit.

Scored on **ROC AUC**.

## Results

Three models, each tuned by 5-fold `GridSearchCV` on ROC AUC, evaluated on a
held-out test split:

| Model | CV ROC AUC | Test ROC AUC | Test accuracy |
|---|---|---|---|
| Logistic regression (L1/L2, tuned C) | 0.9419 | 0.9425 | 0.915 |
| Decision tree (tuned depth, min split) | 0.9458 | 0.9474 | 0.923 |
| **Random forest** (tuned trees, depth, min split) | **0.9544** | **0.9538** | 0.911 |

CV and test scores agree to within ~0.001 on all three, so the tuning is not
overfitting the validation folds.

Worth noting: the random forest wins clearly on ROC AUC but has the *lowest*
accuracy of the three. The classes are imbalanced, so accuracy rewards
predicting the majority class and is the wrong lens here — ROC AUC is the
competition metric for good reason.

## Features

The dataset is the classic bank marketing schema:

- **Demographics**: age, job, marital status, education
- **Financial**: account balance, housing loan, personal loan, credit default
- **Campaign**: contact type, day, month, call duration, number of contacts
- **History**: days since last contact (`pdays`), previous contacts, previous outcome

## Layout

| File | What |
|---|---|
| `01_data_and_EDA.ipynb` | Exploratory analysis, 42 figures |
| `02-preparation-and-model.ipynb` | Preprocessing pipeline, three tuned models, evaluation |

## Reproducing

Competition data is not committed (`train.csv` alone is 63 MB). Download
`train.csv`, `test.csv` and `sample_submission.csv` from the competition's
Data tab into the repository root, then run the notebooks in order.

## Scope

This is an early, straightforward pass: scikit-learn pipelines, grid search,
three model families. It does not use gradient boosting, feature engineering
beyond the categorical encoding, or a frozen cross-validation scheme shared
across models.

For a more developed treatment of the same kind of problem, see
[stellar-class-s6e6-ensemble](https://github.com/szymkas02-coder/stellar-class-s6e6-ensemble)
and [kaggle-f1-pitstops-s6e5](https://github.com/szymkas02-coder/kaggle-f1-pitstops-s6e5),
which use frozen CV, out-of-fold prediction matrices and multi-model blending.
