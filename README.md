# Credit Risk Scorecard

## Business Question
Can we predict whether a loan applicant will default, and use that 
prediction to make a cost-aware approval decision that maximises 
expected profit while managing bad-debt risk?

## Dataset
Home Credit Default Risk — available on Kaggle.  
Raw data files are not committed to this repo. Download from:  
https://www.kaggle.com/c/home-credit-default-risk/data  
Place files in the `/data` folder.

## Approach (CRISP-DM)
1. **Business Understanding** — Define default, approval threshold, 
   and cost matrix (false negative vs false positive cost)
2. **Data Understanding** — Explore application, bureau, and 
   previous loan tables
3. **Data Preparation** — Join tables, engineer domain-aware features, 
   handle class imbalance
4. **Modelling** — Build both a WOE/logistic scorecard and a 
   LightGBM model, compare performance
5. **Evaluation** — KS statistic, Gini coefficient, calibration curve, 
   cost-based threshold selection
6. **Deployment** — Streamlit app for interactive scoring (planned)

## Key Differentiators
- WOE scorecard alongside gradient boosting (interpretability vs performance)
- Probability calibration check
- Cost-sensitive approval threshold — model output translated into 
  a business decision with expected profit per 1000 applicants

## Status
🔵 In progress