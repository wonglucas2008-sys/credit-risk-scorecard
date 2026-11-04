# Credit Risk Scorecard
Predicting loan default and translating predictions into a cost-aware approval decision that maximises expected profit.

## Key Result
At a cost-optimised cutoff of 0.16 (chosen on one half of the held-out test set; theoretical break-even 0.143), the calibrated, gender-free LightGBM earns about **$548 profit per applicant** on the other, untouched half. That is about **$113 per applicant more than approving everyone ($435)**, with an 87.2% approval rate. Profit figures rest on assumed flat costs (see Limitations).

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
0. **Business Understanding** — Define default, approval threshold, and cost matrix
1. **Data Understanding** — Explore application, bureau, and previous loan tables
2. **Data Preparation** — Join tables, engineer domain-aware features, flag the
   DAYS_EMPLOYED anomaly. Missing numeric values are imputed after the train/test
   split, using training medians only
3. **Modelling** — A WOE/logistic scorecard (12 features, IV > 0.02) and a LightGBM
   model, compared like for like
4. **Evaluation** — KS, Gini, isotonic calibration, cost-based threshold chosen and
   reported on separate data, fairness analysis
5. **Deployment** — Streamlit dashboard for scoring applicants and testing cost assumptions

## Models Compared (held-out test set, 61,503 applicants)

| Metric | WOE Scorecard | LightGBM (no gender) |
|--------|--------------|----------------------|
| Features | 12 | 272 |
| AUC | 0.7345 | 0.7679 |
| KS Statistic | 0.3506 | 0.4026 |
| Gini | 0.4691 | 0.5357 |
| Interpretability | High | Low |
| Calibrated | Yes (logistic) | After isotonic fix |

### Is the comparison fair?
The scorecard uses 12 features and LightGBM uses many more, so I added a controlled comparison (test AUC):

| Model | Features | AUC |
|-------|----------|-----|
| WOE scorecard + logistic regression | 12 | 0.7345 |
| LightGBM | same 12 | 0.7490 |
| LightGBM (with gender, for reference) | 275 | 0.7700 |

Of the 0.0355 AUC gap between the scorecard and LightGBM with all features, about 0.0145 comes from the model class and about 0.0210 from the extra features. The scorecard is closer to LightGBM than the headline comparison suggests. All figures come from a single random split.

## Key Findings
- LightGBM outperforms the scorecard on all ranking metrics, but more than half of the gap comes from having more features; the scorecard offers superior interpretability for regulatory needs
- LightGBM is trained with class weights, so raw scores are not true probabilities; isotonic regression fitted on a validation set maps them to calibrated probabilities
- External credit bureau scores (EXT_SOURCE) are the strongest predictors in both models
- Gender appeared as a predictor in LightGBM; removing it cost about 0.002 AUC (0.7700 to 0.7679), so it is excluded from the deployed model for fair-lending compliance
- Information Value must be computed on binned data: `sc.iv()` treats every unique value as its own group and inflates IV for continuous features

## Methodology Notes
- Stratified 80/20 train/test split (`random_state=42`); train is split again 80/20 into train and validation. Validation is used only for LightGBM early stopping and for fitting the calibrator
- The test set is split in half: half A chooses the approval threshold, half B reports profit
- Scorecard features: IV > 0.02 computed from `woebin` bins, with the duplicate DAYS_BIRTH dropped in favour of AGE_YEARS

## Run the dashboard
```
pip install streamlit plotly joblib lightgbm scikit-learn pandas
python -m streamlit run 06_dashboard.py
```
The first launch trains the model (a few minutes) and caches it in `models/`.

## Limitations
- **Assumed costs.** +$1,000 per good loan and -$6,000 per bad loan come from an assumed $10,000 loan, 10% margin and 60% loss given default, not from the data
- **Flat loan size.** Every loan is treated as the same size
- **Reject inference.** Only approved loans have known outcomes, so the model is trained on a filtered population
- **Random split.** No out-of-time validation; results come from a single split and seed
- **Fairness.** Removing the explicit attribute does not rule out proxy discrimination