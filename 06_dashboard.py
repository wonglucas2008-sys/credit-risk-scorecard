"""
06 — Deployment: Streamlit Dashboard
Credit Risk Scorecard Project (CRISP-DM Phase 6)

Run from the project root:
    python -m streamlit run 06_dashboard.py

First launch trains the gender-free LightGBM model, fits the isotonic
calibrator and caches everything to /models (about 1-3 minutes).
Later launches load from disk in seconds. Delete /models to retrain.

Reproduces the exact splits used in notebooks 03-05:
    80/20 stratified train/test (random_state=42); train is split again
    80/20 into train / validation (early stopping + calibrator); the test
    set is split 50/50 into half A (chooses the threshold) and half B
    (reports profit). Imputation uses train medians only.
"""

import os
import re

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH = "data/processed/app_clean.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "lgbm_fair.txt")
ARTIFACT_PATH = os.path.join(MODEL_DIR, "dashboard_artifacts.joblib")

SCORECARD_METRICS = {"AUC": 0.7345, "KS": 0.3506, "Gini": 0.4691}  # notebook 03

# Categorical groups exposed in the applicant form (one-hot prefixes)
CAT_GROUPS = {
    "Education": "NAME_EDUCATION_TYPE_",
    "Income type": "NAME_INCOME_TYPE_",
    "Contract type": None,  # label-encoded binary, handled separately
}

COLORS = {"good": "#2E7D6B", "bad": "#C0392B", "neutral": "#5B6C8F", "accent": "#E0A526"}

st.set_page_config(page_title="Credit Risk Scorecard", page_icon="💳", layout="wide")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def clean_cols(cols):
    return [re.sub(r"[^A-Za-z0-9_]", "_", c) for c in cols]


def ks_stat(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def money(x):
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1e6:
        return f"{sign}${x / 1e6:,.2f}M"
    if x >= 1e3:
        return f"{sign}${x / 1e3:,.1f}K"
    return f"{sign}${x:,.0f}"


def gender_labels(X):
    if "CODE_GENDER_F" in X.columns:
        f = X["CODE_GENDER_F"].values == 1
        m = X["CODE_GENDER_M"].values == 1 if "CODE_GENDER_M" in X.columns else ~f
        return np.where(f, "Female", np.where(m, "Male", "Other"))
    return np.full(len(X), "Unknown")


# ---------------------------------------------------------------------------
# Training / artifact building (runs once, then cached to disk)
# ---------------------------------------------------------------------------
def build_artifacts():
    app = pd.read_csv(DATA_PATH)
    X = app.drop(columns=["TARGET", "SK_ID_CURR"])
    y = app["TARGET"].astype(int)
    X.columns = clean_cols(X.columns)
    X = X.astype(float)

    # Same splits as notebooks 04-05
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.2, random_state=42, stratify=y_trainval
    )

    # Impute with TRAIN medians only
    train_medians = X_train.median()
    X_train = X_train.fillna(train_medians)
    X_val = X_val.fillna(train_medians)
    X_test = X_test.fillna(train_medians)

    gender_cols = [c for c in X.columns if "GENDER" in c.upper()]
    feats = [c for c in X.columns if c not in gender_cols]

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    params = {
        "objective": "binary", "metric": "auc",
        "learning_rate": 0.05, "num_leaves": 31,
        "scale_pos_weight": neg / pos, "random_state": 42, "verbose": -1,
    }

    def fit(cols):
        d = lgb.Dataset(X_train[cols], label=y_train)
        v = lgb.Dataset(X_val[cols], label=y_val, reference=d)
        return lgb.train(params, d, num_boost_round=500, valid_sets=[v],
                         callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])

    # Production model: protected attribute removed. Early stopping uses VALIDATION.
    model = fit(feats)
    # Reference model with gender, only for the fairness comparison
    model_full = fit(list(X.columns))

    raw_test = model.predict(X_test[feats], num_iteration=model.best_iteration)
    raw_val = model.predict(X_val[feats], num_iteration=model.best_iteration)
    auc_fair = roc_auc_score(y_test, raw_test)
    auc_full = roc_auc_score(y_test, model_full.predict(
        X_test[list(X.columns)], num_iteration=model_full.best_iteration))

    # Calibrator fitted on VALIDATION; bounds avoid exact 0 and 1
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999).fit(raw_val, y_val)

    # Test set -> half A (chooses threshold) and half B (reports profit).
    # Splitting an index array reproduces notebook 05's split exactly
    # and lets us carry gender along for the fairness audit.
    y_t = y_test.values
    idx_A, idx_B = train_test_split(
        np.arange(len(y_t)), test_size=0.5, random_state=42, stratify=y_t
    )
    cal_A = iso.predict(raw_test[idx_A])
    raw_B = raw_test[idx_B]
    cal_B = iso.predict(raw_B)

    importance = pd.DataFrame({
        "feature": feats,
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)

    cat_options = {}
    for label, prefix in CAT_GROUPS.items():
        if prefix:
            cols = [c for c in feats if c.startswith(prefix)]
            if cols:
                cat_options[label] = cols

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_model(MODEL_PATH)
    artifacts = {
        "feats": feats,
        "medians": train_medians[feats],
        "iso": iso,
        "y_test": y_t, "raw_test": raw_test,
        # "val" keys now hold test half B (the reporting half) so the app code is unchanged
        "y_val": y_t[idx_B], "raw_val": raw_B, "cal_val": cal_B,
        "y_A": y_t[idx_A], "cal_A": cal_A,
        "gender_val": gender_labels(X_test.iloc[idx_B]),
        "auc_fair": auc_fair, "auc_full": auc_full,
        "importance": importance,
        "cat_options": cat_options,
        "n_train": len(X_train),
        "default_rate": float(y.mean()),
    }
    joblib.dump(artifacts, ARTIFACT_PATH)
    return model, artifacts


@st.cache_resource(show_spinner=False)
def load_everything():
    if os.path.exists(ARTIFACT_PATH) and os.path.exists(MODEL_PATH):
        return lgb.Booster(model_file=MODEL_PATH), joblib.load(ARTIFACT_PATH)
    if not os.path.exists(DATA_PATH):
        st.error(f"Could not find `{DATA_PATH}`. Run notebook 02 first, "
                 "then launch the app from the project root.")
        st.stop()
    with st.spinner("First run: training LightGBM and calibrating (1-3 minutes)..."):
        return build_artifacts()


@st.cache_data(show_spinner=False)
def profit_curve(y, p, profit_good, loss_bad):
    thresholds = np.round(np.arange(0.01, 1.00, 0.01), 2)
    rows = []
    for t in thresholds:
        approved = p < t
        good = int((approved & (y == 0)).sum())
        bad = int((approved & (y == 1)).sum())
        rows.append({
            "threshold": t, "approval_rate": approved.mean(),
            "approved_good": good, "approved_bad": bad,
            "bad_rate_approved": bad / max(good + bad, 1),
            "profit": good * profit_good - bad * loss_bad,
        })
    return pd.DataFrame(rows)


model, A = load_everything()
y_val, cal_val = A["y_val"], A["cal_val"]

# ---------------------------------------------------------------------------
# Sidebar: business assumptions drive every tab
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Business assumptions")
    loan = st.number_input("Average loan ($)", 1_000, 100_000, 10_000, step=1_000)
    margin = st.slider("Interest margin", 0.01, 0.30, 0.10, 0.01, format="%.2f")
    lgd = st.slider("Loss given default", 0.10, 1.00, 0.60, 0.05, format="%.2f")
    profit_good, loss_bad = loan * margin, loan * lgd
    breakeven = profit_good / (profit_good + loss_bad)

    pg_txt = money(profit_good).replace("$", "\\$")
    lb_txt = money(loss_bad).replace("$", "\\$")
    st.caption(f"Good loan: **+{pg_txt}** · Bad loan: **-{lb_txt}**  \n"
               f"Cost ratio {loss_bad / profit_good:.1f}x · break-even PD **{breakeven:.3f}**")

    curve = profit_curve(y_val, cal_val, profit_good, loss_bad)            # half B: reported
    curve_A = profit_curve(A["y_A"], A["cal_A"], profit_good, loss_bad)    # half A: chooses
    best = curve_A.loc[curve_A["profit"].idxmax()]

    st.divider()
    st.header("Approval threshold")
    mode = st.radio("Threshold", ["Profit-optimal", "Manual"], horizontal=True,
                    label_visibility="collapsed")
    if mode == "Manual":
        threshold = st.slider("Approve if PD below", 0.01, 0.99,
                              float(best["threshold"]), 0.01)
    else:
        threshold = float(best["threshold"])
        st.metric("Optimal cutoff", f"{threshold:.2f}")

    st.divider()
    st.caption(f"Model: LightGBM, gender removed, isotonic-calibrated. "
               f"Trained on {A['n_train']:,} applications. "
               f"Cutoff chosen on test half A; profit reported on the {len(y_val):,}-row test half B.")

row = curve.loc[(curve["threshold"] - threshold).abs().idxmin()]
n_val = len(y_val)
profit_all = int((y_val == 0).sum() * profit_good - (y_val == 1).sum() * loss_bad)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Credit Risk Scorecard")
st.caption("Home Credit Default Risk · predicting default and turning it into "
           "a cost-aware approval decision")

tab_overview, tab_score, tab_profit, tab_diag, tab_fair = st.tabs(
    ["Overview", "Score an applicant", "Threshold & profit", "Model diagnostics", "Fairness"]
)

# ---------------------------------------------------------------------------
# Tab 1: Overview
# ---------------------------------------------------------------------------
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Profit per applicant", money(row["profit"] / n_val))
    c2.metric("Uplift vs approve-all", money(row["profit"] - profit_all),
              f"{(row['profit'] - profit_all) / abs(profit_all):+.0%}" if profit_all else None)
    c3.metric("Approval rate", f"{row['approval_rate']:.1%}")
    c4.metric("Default rate among approved", f"{row['bad_rate_approved']:.2%}",
              f"{row['bad_rate_approved'] - A['default_rate']:+.2%} vs portfolio",
              delta_color="inverse")

    st.subheader("Model comparison")
    auc = A["auc_fair"]
    lgbm = {"AUC": auc, "KS": ks_stat(A["y_test"], A["raw_test"]), "Gini": 2 * auc - 1}
    comp = pd.DataFrame({
        "Metric": ["AUC", "KS statistic", "Gini", "Interpretability", "Calibration"],
        "WOE scorecard": [f"{SCORECARD_METRICS['AUC']:.4f}", f"{SCORECARD_METRICS['KS']:.4f}",
                          f"{SCORECARD_METRICS['Gini']:.4f}", "High", "Native (logistic)"],
        "LightGBM (no gender)": [f"{lgbm['AUC']:.4f}", f"{lgbm['KS']:.4f}",
                                 f"{lgbm['Gini']:.4f}", "Low", "Isotonic fix"],
    })
    left, right = st.columns([3, 2])
    left.dataframe(comp, hide_index=True, width="stretch")

    strat = pd.DataFrame({
        "Strategy": ["Approve nobody", "Approve everyone", f"Model (cutoff {threshold:.2f})"],
        "Profit": [0, profit_all, row["profit"]],
    })
    fig = go.Figure(go.Bar(
        x=strat["Profit"], y=strat["Strategy"], orientation="h",
        marker_color=[COLORS["neutral"], COLORS["neutral"], COLORS["good"]],
        text=[money(v) for v in strat["Profit"]], textposition="auto",
    ))
    fig.update_layout(title="Profit on held-out test half B", height=260,
                      margin=dict(l=10, r=10, t=40, b=10), xaxis_title="Profit ($)")
    right.plotly_chart(fig, width="stretch")

    st.info(
        f"With these assumptions a bad loan costs {loss_bad / profit_good:.1f}x a good loan's profit, "
        f"so the theoretical break-even PD is {breakeven:.3f}. Because the probabilities are "
        f"calibrated, the empirical profit-optimal cutoff ({best['threshold']:.2f}) lands right "
        f"next to it, which is a good sanity check on the calibration."
    )

# ---------------------------------------------------------------------------
# Tab 2: Score an applicant
# ---------------------------------------------------------------------------
with tab_score:
    feats, med = A["feats"], A["medians"]
    st.caption("Fields you don't set use the training-set median. "
               "Engineered ratios are recomputed from your inputs, as in notebook 02.")

    def num(col, label, lo, hi, step, fmt=None, scale=1.0):
        default = float(med.get(col, lo)) / scale
        default = min(max(default, lo), hi)
        return st.number_input(label, float(lo), float(hi), float(default),
                               step=float(step), format=fmt)

    with st.form("applicant"):
        a, b, c = st.columns(3)
        with a:
            st.markdown("**External credit scores**")
            ext1 = st.slider("EXT_SOURCE_1", 0.0, 1.0, float(med.get("EXT_SOURCE_1", 0.5)), 0.01)
            ext2 = st.slider("EXT_SOURCE_2", 0.0, 1.0, float(med.get("EXT_SOURCE_2", 0.5)), 0.01)
            ext3 = st.slider("EXT_SOURCE_3", 0.0, 1.0, float(med.get("EXT_SOURCE_3", 0.5)), 0.01)
        with b:
            st.markdown("**Applicant**")
            age = num("DAYS_BIRTH", "Age (years)", 21, 69, 1, "%.0f", scale=365)
            yrs_emp = num("DAYS_EMPLOYED", "Years employed", 0, 50, 1, "%.1f", scale=365)
            fam = num("CNT_FAM_MEMBERS", "Family members", 1, 20, 1, "%.0f")
            cat_choice = {}
            for label, cols in A["cat_options"].items():
                names = [re.sub(r"\s+", " ", c.split("_", 3)[-1].replace("_", " ")).strip() for c in cols]
                current = int(np.argmax([med.get(c, 0) for c in cols]))
                pick = st.selectbox(label, names, index=current)
                cat_choice[label] = cols[names.index(pick)]
        with c:
            st.markdown("**Loan and history**")
            income = num("AMT_INCOME_TOTAL", "Annual income", 10_000, 10_000_000, 5_000, "%.0f")
            credit = num("AMT_CREDIT", "Credit amount", 10_000, 5_000_000, 10_000, "%.0f")
            annuity = num("AMT_ANNUITY", "Annuity", 1_000, 300_000, 1_000, "%.0f")
            goods = num("AMT_GOODS_PRICE", "Goods price", 10_000, 5_000_000, 10_000, "%.0f")
            refused = num("PREV_REFUSED_COUNT", "Previous refused applications", 0, 50, 1, "%.0f")
        submitted = st.form_submit_button("Score applicant", type="primary")

    x = med.copy()
    updates = {
        "EXT_SOURCE_1": ext1, "EXT_SOURCE_2": ext2, "EXT_SOURCE_3": ext3,
        "DAYS_BIRTH": age * 365, "AGE_YEARS": age,
        "DAYS_EMPLOYED": yrs_emp * 365, "CNT_FAM_MEMBERS": fam,
        "AMT_INCOME_TOTAL": income, "AMT_CREDIT": credit,
        "AMT_ANNUITY": annuity, "AMT_GOODS_PRICE": goods,
        "PREV_REFUSED_COUNT": refused,
        "CREDIT_INCOME_RATIO": credit / income,
        "ANNUITY_INCOME_RATIO": annuity / income,
        "CREDIT_GOODS_RATIO": credit / goods,
        "INCOME_PER_PERSON": income / max(fam, 1),
        "EMPLOYED_TO_AGE_RATIO": (yrs_emp * 365) / (age * 365),
    }
    for k, v in updates.items():
        if k in x.index:
            x[k] = v
    for label, cols in A["cat_options"].items():
        x[cols] = 0.0
        x[cat_choice[label]] = 1.0

    X_one = pd.DataFrame([x[feats].values], columns=feats)
    raw = float(model.predict(X_one)[0])
    pd_cal = float(A["iso"].predict([raw])[0])
    ev = (1 - pd_cal) * profit_good - pd_cal * loss_bad
    approve = pd_cal < threshold

    g, d = st.columns([2, 3])
    with g:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=pd_cal * 100,
            number={"suffix": "%", "valueformat": ".1f"},
            title={"text": "Calibrated probability of default"},
            gauge={
                "axis": {"range": [0, max(40, pd_cal * 100 + 5)]},
                "bar": {"color": COLORS["bad"] if not approve else COLORS["good"]},
                "threshold": {"line": {"color": "black", "width": 3}, "value": threshold * 100},
                "steps": [{"range": [0, threshold * 100], "color": "#E3F1EC"}],
            },
        ))
        gauge.update_layout(height=280, margin=dict(l=20, r=20, t=60, b=10))
        st.plotly_chart(gauge, width="stretch")
        if approve:
            st.success(f"**APPROVE** · PD {pd_cal:.1%} is below the {threshold:.2f} cutoff")
        else:
            st.error(f"**REJECT** · PD {pd_cal:.1%} is at or above the {threshold:.2f} cutoff")
        st.metric("Expected value of approving", money(ev))

    with d:
        contrib = model.predict(X_one, pred_contrib=True)[0][:-1]
        top = (pd.DataFrame({"feature": feats, "contrib": contrib})
               .assign(abs=lambda t: t["contrib"].abs())
               .nlargest(12, "abs").sort_values("contrib"))
        fig = go.Figure(go.Bar(
            x=top["contrib"], y=top["feature"], orientation="h",
            marker_color=np.where(top["contrib"] > 0, COLORS["bad"], COLORS["good"]),
        ))
        fig.update_layout(
            title="What drove this score (LightGBM contributions, log-odds)",
            height=420, margin=dict(l=10, r=10, t=40, b=10),
            xaxis_title="← lowers default risk · raises default risk →",
        )
        st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------------------
# Tab 3: Threshold & profit
# ---------------------------------------------------------------------------
with tab_profit:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve["threshold"], y=curve["profit"], name="Model",
                             line=dict(color=COLORS["neutral"], width=3)))
    fig.add_hline(y=profit_all, line_dash="dot", line_color="gray",
                  annotation_text="Approve everyone", annotation_position="bottom right")
    fig.add_vline(x=best["threshold"], line_dash="dash", line_color=COLORS["bad"],
                  annotation_text=f"Optimal {best['threshold']:.2f}")
    fig.add_vline(x=breakeven, line_dash="dot", line_color=COLORS["accent"],
                  annotation_text=f"Break-even {breakeven:.3f}", annotation_position="bottom left")
    if mode == "Manual":
        fig.add_vline(x=threshold, line_color="black",
                      annotation_text=f"Selected {threshold:.2f}", annotation_position="top left")
    fig.update_layout(title="Profit vs approval threshold", height=420,
                      xaxis_title="Approve if calibrated PD below", yaxis_title="Profit ($)",
                      margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, width="stretch")

    l, r = st.columns(2)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=curve["threshold"], y=curve["approval_rate"],
                              name="Approval rate", line=dict(color=COLORS["good"])))
    fig2.add_trace(go.Scatter(x=curve["threshold"], y=curve["bad_rate_approved"],
                              name="Default rate among approved", line=dict(color=COLORS["bad"])))
    fig2.add_vline(x=threshold, line_dash="dash", line_color="black")
    fig2.update_layout(title="Volume vs risk trade-off", height=340, yaxis_tickformat=".0%",
                       xaxis_title="Threshold", margin=dict(l=10, r=10, t=50, b=10),
                       legend=dict(orientation="h", y=-0.3))
    l.plotly_chart(fig2, width="stretch")

    with r:
        st.markdown(f"**At cutoff {threshold:.2f}** (test half B, {n_val:,} applicants)")
        tbl = pd.DataFrame({
            "": ["Approved", "Rejected"],
            "Would repay": [row["approved_good"], int((y_val == 0).sum() - row["approved_good"])],
            "Would default": [row["approved_bad"], int((y_val == 1).sum() - row["approved_bad"])],
        })
        st.dataframe(tbl, hide_index=True, width="stretch")
        st.metric("Total profit", money(row["profit"]))
        st.metric("Profit per 1,000 applicants", money(row["profit"] / n_val * 1000))

    with st.expander("Full threshold table"):
        st.dataframe(curve.style.format({
            "approval_rate": "{:.1%}", "bad_rate_approved": "{:.2%}", "profit": "${:,.0f}",
            "threshold": "{:.2f}",
        }), hide_index=True, width="stretch")

# ---------------------------------------------------------------------------
# Tab 4: Model diagnostics
# ---------------------------------------------------------------------------
with tab_diag:
    l, r = st.columns(2)

    fpr, tpr, _ = roc_curve(A["y_test"], A["raw_test"])
    ks_i = int(np.argmax(tpr - fpr))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"LightGBM (AUC {A['auc_fair']:.3f})",
                             line=dict(color=COLORS["neutral"], width=3)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random",
                             line=dict(color="gray", dash="dot")))
    fig.add_trace(go.Scatter(x=[fpr[ks_i], fpr[ks_i]], y=[fpr[ks_i], tpr[ks_i]],
                             name=f"KS {tpr[ks_i] - fpr[ks_i]:.3f}", mode="lines",
                             line=dict(color=COLORS["bad"], width=3)))
    fig.update_layout(title="ROC curve (test set)", height=420,
                      xaxis_title="False positive rate", yaxis_title="True positive rate",
                      margin=dict(l=10, r=10, t=50, b=10), legend=dict(x=0.45, y=0.1))
    l.plotly_chart(fig, width="stretch")

    t_raw, p_raw = calibration_curve(y_val, A["raw_val"], n_bins=10)
    t_cal, p_cal = calibration_curve(y_val, cal_val, n_bins=10)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Perfect",
                             line=dict(color="gray", dash="dot")))
    fig.add_trace(go.Scatter(x=p_raw, y=t_raw, name="Raw (class-weighted)",
                             mode="lines+markers", line=dict(color=COLORS["bad"])))
    fig.add_trace(go.Scatter(x=p_cal, y=t_cal, name="Isotonic calibrated",
                             mode="lines+markers", line=dict(color=COLORS["good"])))
    fig.update_layout(title="Calibration (test half B)", height=420,
                      xaxis_title="Mean predicted probability", yaxis_title="Actual default rate",
                      margin=dict(l=10, r=10, t=50, b=10), legend=dict(x=0.02, y=0.98))
    r.plotly_chart(fig, width="stretch")

    l, r = st.columns(2)
    fig = go.Figure()
    for lbl, val, col in [("Repaid", 0, COLORS["good"]), ("Defaulted", 1, COLORS["bad"])]:
        fig.add_trace(go.Histogram(x=cal_val[y_val == val], name=lbl, marker_color=col,
                                   opacity=0.6, histnorm="probability", nbinsx=50))
    fig.add_vline(x=threshold, line_dash="dash", line_color="black",
                  annotation_text=f"Cutoff {threshold:.2f}")
    fig.update_layout(barmode="overlay", title="Calibrated PD by outcome", height=420,
                      xaxis_title="Calibrated PD", yaxis_title="Share of group",
                      margin=dict(l=10, r=10, t=50, b=10))
    l.plotly_chart(fig, width="stretch")

    imp = A["importance"].head(20).iloc[::-1]
    fig = go.Figure(go.Bar(x=imp["importance"], y=imp["feature"], orientation="h",
                           marker_color=COLORS["neutral"]))
    fig.update_layout(title="Top 20 features by gain", height=420,
                      margin=dict(l=10, r=10, t=50, b=10))
    r.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------------------
# Tab 5: Fairness
# ---------------------------------------------------------------------------
with tab_fair:
    c1, c2, c3 = st.columns(3)
    c1.metric("AUC with gender", f"{A['auc_full']:.4f}")
    c2.metric("AUC without gender", f"{A['auc_fair']:.4f}",
              f"{A['auc_fair'] - A['auc_full']:+.4f}")
    c3.metric("Deployed model", "Gender removed")

    st.subheader(f"Outcomes by gender at cutoff {threshold:.2f}")
    st.caption("The model never sees gender. It is used here only to audit outcomes "
               "for indirect (proxy) discrimination.")
    g = A["gender_val"]
    approved = cal_val < threshold
    rows = []
    for grp in [x for x in ["Female", "Male", "Other", "Unknown"] if (g == x).sum() >= 30]:
        m = g == grp
        rows.append({
            "Group": grp, "Applicants": int(m.sum()),
            "Actual default rate": y_val[m].mean(),
            "Approval rate": approved[m].mean(),
            "Default rate among approved": y_val[m & approved].mean() if (m & approved).any() else np.nan,
            "Mean PD": cal_val[m].mean(),
        })
    fair = pd.DataFrame(rows)

    if len(fair) >= 2:
        di = fair["Approval rate"].min() / fair["Approval rate"].max()
        l, r = st.columns([3, 2])
        l.dataframe(fair.style.format({
            "Applicants": "{:,}", "Actual default rate": "{:.2%}", "Approval rate": "{:.1%}",
            "Default rate among approved": "{:.2%}", "Mean PD": "{:.3f}",
        }), hide_index=True, width="stretch")
        with r:
            st.metric("Disparate impact ratio", f"{di:.3f}",
                      "passes four-fifths rule" if di >= 0.8 else "fails four-fifths rule",
                      delta_color="normal" if di >= 0.8 else "inverse")
            st.caption("Lowest group approval rate ÷ highest. Below 0.80 is the common "
                       "regulatory flag for adverse impact.")

        fig = go.Figure(go.Bar(
            x=fair["Group"], y=fair["Approval rate"], marker_color=COLORS["neutral"],
            text=[f"{v:.1%}" for v in fair["Approval rate"]], textposition="outside",
        ))
        fig.update_layout(title="Approval rate by gender", height=320, yaxis_tickformat=".0%",
                          yaxis_range=[0, 1.05], margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, width="stretch")
    else:
        st.warning("Gender columns were not found in the validation data, "
                   "so outcomes can't be audited by group.")

    st.info(
        "Removing the explicit attribute is necessary but not sufficient. "
        "If approval rates still differ between groups, compare them with the actual "
        "default rates: a gap that tracks real risk differences is different from one "
        "driven by proxies such as occupation or income type. Re-run this check "
        "whenever the threshold or assumptions change."
    )
