# Data Cleaning Notes

## Target Definition
- **Target**: 1 = Bad (Charged Off or Default), 0 = Good (Fully Paid)
- Dropped all rows with intermediate statuses (Current, In Grace Period, etc.) because their final outcome is unknown.

## Leakage Columns Dropped
The following columns were identified as post-outcome leakage and dropped:
- `total_rec_prncp`, `total_pymnt`, `total_pymnt_inv`, `recoveries`, `last_pymnt_d`, `last_pymnt_amnt`, `collection_recovery_fee`, `out_prncp`, `out_prncp_inv`, `next_pymnt_d`, `total_rec_int`, `total_rec_late_fee`, `last_credit_pull_d`, `settlement_status`, `debt_settlement_flag`, `hardship_flag`
*Reasoning*: These represent payments made or actions taken *after* the loan was originated, which the model would not have access to at the time of prediction.

## Missing Values Handling
- **Informative Missingness**: Columns like `mths_since_last_delinq` (months since last delinquency). A missing value often means "never delinquent", which is a strong signal. For these, a `{column}_was_missing` flag was created, and the original column was imputed with the median.
- **MAR Features (Numeric)**: Imputed with the column median to be robust against outliers.
- **MAR Features (Categorical)**: Imputed with the column mode.

## Dates
- `issue_d` and `earliest_cr_line` were parsed as datetimes for future time-based feature engineering.
