# Credit Risk Scorecard
Predicting loan default and translating predictions into a cost-aware approval decision that maximises expected profit.

## Key Result
At the cost-optimised threshold (0.14), the calibrated LightGBM model generates approximately **$554 profit per applicant**, about **$3.67M more than an approve-all baseline** on the validation set.

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
5. **Evaluation** — KS, Gini, calibration with isotonic fix, cost-based threshold, fairness analysis
6. **Deployment** — Streamlit app for interactive scoring (planned)

## Models Compared

| Metric | WOE Scorecard | LightGBM |
|--------|--------------|----------|
| AUC | 0.7391 | 0.7712 |
| KS Statistic | 0.3558 | 0.4025 |
| Gini | 0.4783 | 0.5424 |
| Interpretability | High | Low |
| Calibrated | Yes (linear) | After isotonic fix |

## Key Differentiators
- WOE scorecard alongside gradient boosting (interpretability vs performance)
- Probability calibration check
- Cost-sensitive approval threshold, translating model output into a business decision with expected profit per 1,000 applicants

## Key Findings
- LightGBM outperforms the scorecard on all ranking metrics, but the scorecard offers superior interpretability for regulatory needs
- Raw LightGBM probabilities were poorly calibrated due to class weighting; isotonic regression corrected this
- External credit bureau scores (EXT_SOURCE) are the strongest predictors
- Gender appeared as a predictor; removing it cost only 0.0010 AUC, so it should be removed for fair-lending compliance

## Limitations
- Median imputation used for high-missingness columns
- Random rather than time-based train/test split
- Fairness addressed only by removing the explicit attribute; proxy discrimination not yet tested