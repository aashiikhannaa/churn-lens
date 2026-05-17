import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Lens",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global CSS — dark theme ────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.main { background: #111110; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }
#MainMenu, footer, header { visibility: hidden; }

[data-testid="stSidebar"] {
    background: #1a1a18;
    border-right: 1px solid #2a2a28;
}

.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.75rem;
    color: #f0ede6;
    margin-bottom: 0.2rem;
    line-height: 1.2;
}
.section-sub {
    font-size: 0.88rem;
    color: #666660;
    margin-bottom: 1.5rem;
}
.metric-card {
    background: #1a1a18;
    border: 1px solid #2a2a28;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
}
.metric-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: #555550;
    margin-bottom: 0.3rem;
}
.metric-value { font-size: 1.75rem; font-weight: 600; color: #f0ede6; line-height: 1; }
.metric-caption { font-size: 0.75rem; color: #555550; margin-top: 0.3rem; }

.risk-high {
    display:inline-block; background:#2d1515; color:#f87171;
    border:1px solid #5a2020; border-radius:20px;
    padding:5px 16px; font-size:0.8rem; font-weight:600; letter-spacing:0.04em;
}
.risk-medium {
    display:inline-block; background:#2a1f0e; color:#fbbf24;
    border:1px solid #5a3a10; border-radius:20px;
    padding:5px 16px; font-size:0.8rem; font-weight:600; letter-spacing:0.04em;
}
.risk-low {
    display:inline-block; background:#0e2318; color:#4ade80;
    border:1px solid #1a4a2a; border-radius:20px;
    padding:5px 16px; font-size:0.8rem; font-weight:600; letter-spacing:0.04em;
}

.insight-card {
    background:#1a1a18; border:1px solid #2a2a28;
    border-radius:10px; padding:1rem 1.25rem; margin-bottom:1.25rem;
}
.insight-title { font-weight:600; font-size:0.88rem; color:#d0cdc6; margin-bottom:0.4rem; }
.insight-body { font-size:0.83rem; color:#666660; line-height:1.7; }

.soft-divider { border:none; border-top:1px solid #2a2a28; margin:1.25rem 0; }
.learn-body { font-size:0.88rem; color:#888880; line-height:1.8; }
</style>
""", unsafe_allow_html=True)

# ── Matplotlib dark style ──────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#1a1a18',
    'axes.facecolor':   '#1a1a18',
    'axes.edgecolor':   '#2a2a28',
    'axes.labelcolor':  '#666660',
    'xtick.color':      '#555550',
    'ytick.color':      '#555550',
    'text.color':       '#888880',
})

# ── Load artifacts ─────────────────────────────────────────────────────────────
model   = joblib.load('model.pkl')
scaler  = joblib.load('scaler.pkl')
columns = joblib.load('columns.pkl')

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:0.75rem 0 1.25rem 0;'>
        <div style='font-family:DM Serif Display,serif;font-size:1.4rem;color:#f0ede6;'>🔭 Churn Lens</div>
        <div style='font-size:0.75rem;color:#555550;margin-top:0.2rem;'>Customer Churn Predictor</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("Navigate", ["Predict", "Insights", "Learn"],
                    label_visibility="collapsed")

    st.markdown("<hr style='border-top:1px solid #2a2a28;margin:1.25rem 0'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.82rem;color:#babaab;line-height:1.8;'>
        Logistic Regression · Telco Dataset<br>
        AUC-ROC <strong style='color:#888880'>0.828</strong> ·
        Recall <strong style='color:#888880'>0.743</strong>
        <br>
        <br>
    </div>
""", unsafe_allow_html=True)
    
    st.markdown("<hr style='border-top:1px solid #2a2a28;margin:1.25rem 0'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='position:absolute;bottom:1.8rem;font-size:0.85rem;color:#8c8c85;'>
        Built by <b>Aashi Khanna</b> 
    </div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
if page == "Predict":
    st.markdown('<div class="section-title">Predict Churn</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Enter customer details across the tabs, then click Analyse.</div>', unsafe_allow_html=True)

    col_form, col_gap, col_result = st.columns([1.1, 0.08, 1.4])

    with col_form:
        t1, t2, t3 = st.tabs(["Account & Charges", "Services", "Demographics"])

        with t1:
            tenure            = st.slider("Tenure (months)", 0, 72, 12)
            monthly_charges   = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
            total_charges     = st.number_input("Total Charges ($)", 0.0, 9000.0,
                                                 float(tenure * monthly_charges), step=10.0)
            contract          = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            payment_method    = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

        with t2:
            internet_service  = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            phone_service     = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines    = st.selectbox("Multiple Lines", ["Yes", "No"])
            online_security   = st.selectbox("Online Security", ["Yes", "No"])
            online_backup     = st.selectbox("Online Backup", ["Yes", "No"])
            device_protection = st.selectbox("Device Protection", ["Yes", "No"])
            tech_support      = st.selectbox("Tech Support", ["Yes", "No"])
            streaming_tv      = st.selectbox("Streaming TV", ["Yes", "No"])
            streaming_movies  = st.selectbox("Streaming Movies", ["Yes", "No"])

        with t3:
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            gender         = st.selectbox("Gender", ["Male", "Female"])
            partner        = st.selectbox("Partner", ["Yes", "No"])
            dependents     = st.selectbox("Dependents", ["Yes", "No"])

        predict_btn = st.button("Analyse Customer →", use_container_width=True)

    with col_result:
        if predict_btn:
            # ── Build input ────────────────────────────────────────────────
            input_dict = {
                'SeniorCitizen':         1 if senior_citizen == "Yes" else 0,
                'Partner':               1 if partner == "Yes" else 0,
                'Dependents':            1 if dependents == "Yes" else 0,
                'tenure':                tenure,
                'PhoneService':          1 if phone_service == "Yes" else 0,
                'MultipleLines':         1 if multiple_lines == "Yes" else 0,
                'OnlineSecurity':        1 if online_security == "Yes" else 0,
                'OnlineBackup':          1 if online_backup == "Yes" else 0,
                'DeviceProtection':      1 if device_protection == "Yes" else 0,
                'TechSupport':           1 if tech_support == "Yes" else 0,
                'StreamingTV':           1 if streaming_tv == "Yes" else 0,
                'StreamingMovies':       1 if streaming_movies == "Yes" else 0,
                'PaperlessBilling':      1 if paperless_billing == "Yes" else 0,
                'MonthlyCharges':        monthly_charges,
                'TotalCharges':          total_charges,
                'gender':                1 if gender == "Male" else 0,
                'InternetService_Fiber optic': 1 if internet_service == "Fiber optic" else 0,
                'InternetService_No':         1 if internet_service == "No" else 0,
                'Contract_One year':          1 if contract == "One year" else 0,
                'Contract_Two year':          1 if contract == "Two year" else 0,
                'PaymentMethod_Credit card (automatic)': 1 if payment_method == "Credit card (automatic)" else 0,
                'PaymentMethod_Mailed check':            1 if payment_method == "Mailed check" else 0,
                'PaymentMethod_Electronic check':        1 if payment_method == "Electronic check" else 0,
            }

            input_df = pd.DataFrame([input_dict])
            input_df = input_df.reindex(columns=columns, fill_value=0)
            num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
            input_df[num_cols] = scaler.transform(input_df[num_cols])

            prob     = model.predict_proba(input_df)[0][1]
            prob_pct = prob * 100

            if prob >= 0.7:
                risk_label, risk_class, bar_color = "High Risk",   "risk-high",   "#f87171"
                risk_msg = "Strong churn signals detected. Recommend an immediate retention call or targeted discount — especially on monthly charges."
            elif prob >= 0.4:
                risk_label, risk_class, bar_color = "Medium Risk", "risk-medium", "#fbbf24"
                risk_msg = "Moderate churn signals. Consider a proactive check-in or a loyalty offer before the next billing cycle."
            else:
                risk_label, risk_class, bar_color = "Low Risk",    "risk-low",    "#4ade80"
                risk_msg = "Customer appears stable. Standard engagement is sufficient — no immediate action required."

            # Probability bar
            fig_g, ax = plt.subplots(figsize=(5, 1.6))
            ax.barh([0], [100], color='#2a2a28', height=0.35)
            ax.barh([0], [prob_pct], color=bar_color, height=0.35)
            ax.set_xlim(0, 100)
            ax.set_yticks([])
            ax.set_xticks([0, 25, 50, 75, 100])
            ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=8)
            for spine in ax.spines.values():
                spine.set_visible(False)
            label_x = min(prob_pct + 2, 88)
            ax.text(label_x, 0, f'{prob_pct:.1f}%',
                    va='center', fontsize=12, color=bar_color, fontweight='600')
            ax.set_title('Churn Probability', fontsize=9, color='#555550', pad=8, loc='left')
            plt.tight_layout(pad=0.3)
            st.pyplot(fig_g, use_container_width=True)
            plt.close()

            # Risk badge + message
            cb, cm = st.columns([1, 2.5])
            with cb:
                st.markdown(f'<span class="{risk_class}">{risk_label}</span>', unsafe_allow_html=True)
            with cm:
                st.markdown(f'<div style="font-size:0.82rem;color:#666660;line-height:1.6;padding-top:4px;">{risk_msg}</div>',
                            unsafe_allow_html=True)

            st.markdown("<hr class='soft-divider'>", unsafe_allow_html=True)

            # SHAP
            st.markdown("<div style='font-size:0.8rem;font-weight:600;color:#888880;margin-bottom:0.2rem;'>Why this prediction?</div>", unsafe_allow_html=True)
            st.markdown("<div style='font-size:0.78rem;color:#444440;margin-bottom:0.6rem;'>Top features driving this customer's score.</div>", unsafe_allow_html=True)

            background  = pd.DataFrame([np.zeros(len(columns))], columns=columns)
            explainer   = shap.LinearExplainer(model, background)
            shap_vals   = explainer.shap_values(input_df)
            shap_series = pd.Series(shap_vals[0], index=columns)
            shap_top    = shap_series.abs().nlargest(8).index
            shap_plot   = shap_series[shap_top].sort_values()

            fig_s, ax2 = plt.subplots(figsize=(5, 3.2))
            colors = ['#f87171' if v > 0 else '#4ade80' for v in shap_plot.values]
            ax2.barh(shap_plot.index, shap_plot.values, color=colors, height=0.5)
            ax2.axvline(0, color='#3a3a38', linewidth=0.8)
            ax2.set_xlabel('Impact on churn probability', fontsize=8)
            ax2.tick_params(axis='both', labelsize=7.5)
            for spine in ax2.spines.values():
                spine.set_color('#2a2a28')
            red_p   = mpatches.Patch(color='#f87171', label='↑ Increases churn risk')
            green_p = mpatches.Patch(color='#4ade80', label='↓ Reduces churn risk')
            ax2.legend(handles=[red_p, green_p], fontsize=7, loc='upper right', 
                        frameon=True, facecolor='#1a1a18', edgecolor='#2a2a28')
            plt.tight_layout(pad=0.5)
            st.pyplot(fig_s, use_container_width=True)
            plt.close()

            with st.expander("How to read this chart"):
                st.markdown("""
                - **Red bars** push toward churn — longer = stronger signal.
                - **Green bars** push toward loyalty.
                - Shows the **top 8 most influential features** for this specific customer.
                - Powered by **SHAP** (SHapley Additive exPlanations) — feature attribution rooted in game theory.
                """)

        else:
            st.markdown("""
            <div style='margin-top:4rem;text-align:center;'>
                <div style='font-size:2rem;opacity:0.2;margin-bottom:0.75rem;'>🔭</div>
                <div style='color:#555550;font-size:0.9rem;font-weight:500;margin-bottom:0.4rem;'>
                    No prediction yet
                </div>
                <div style='color:#3a3a38;font-size:0.82rem;line-height:1.8;'>
                    Fill in the tabs on the left<br>
                    and click <strong style='color:#555550'>Analyse Customer →</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Insights":
    st.markdown('<div class="section-title">Data Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">What the data tells us about who churns, why, and what drives the model\'s decisions.</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, (label, val, cap) in zip([c1,c2,c3,c4], [
        ("Dataset Size", "7,032", "customers"),
        ("Churn Rate",   "26.5%", "of customers churned"),
        ("Features",     "23",    "after encoding"),
        ("AUC-ROC",      "0.828", "on held-out test set"),
    ]):
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{val}</div>
                <div class="metric-caption">{cap}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='soft-divider'>", unsafe_allow_html=True)

    plots = [
        ("churn_by_contract.png",     "Churn Rate by Contract Type",
         "Month-to-month customers churn at ~43% — over 14× higher than two-year contract holders at ~3%. Longer contracts create commitment friction that strongly protects against churn."),
        ("churn_by_tenure.png",       "Churn Rate by Tenure Group",
         "New customers (0–12 months) are the highest-risk segment. Churn drops sharply after the first year. Customers beyond 48 months are extremely loyal."),
        ("monthly_charges_dist.png",  "Monthly Charges Distribution by Churn",
         "Churners cluster at higher monthly charges. Non-churners concentrate at lower price points. Price pain is the primary driver of departure — confirmed by SHAP."),
        ("roc_curve.png",             "ROC Curve",
         "Plots true positive rate vs false positive rate at every threshold. The further top-left the curve bows, the better. Our AUC of 0.828 indicates strong discriminative ability."),
        ("shap_bar.png",              "SHAP Feature Importance (Global)",
         "MonthlyCharges dominates every other feature by a wide margin. InternetService type is second. Contract type ranks lower than intuition suggests — the financial signal overrides the commitment signal."),
        ("shap_beeswarm.png",         "SHAP Beeswarm — Direction & Magnitude",
         "Each dot is one customer. Red = high feature value, blue = low. Dots right of center push toward churn. High monthly charges is the strongest churn signal. Long tenure is the strongest loyalty signal."),
        ("confusion_matrix.png",      "Confusion Matrix",
         "Breaks predictions into four quadrants. The model prioritises recall — catching churners, accepting some false alarms — because a missed churner costs far more than a wasted call."),
        ("precision_recall_curve.png","Precision-Recall Curve",
         "More informative than ROC for imbalanced datasets. As recall increases (catching more churners), precision drops (more false alarms). This guides the threshold decision."),
    ]

    for i in range(0, len(plots), 2):
        ca, cb = st.columns(2)
        for col, (img, title, caption) in zip([ca, cb], plots[i:i+2]):
            with col:
                st.markdown(f'<div class="insight-title">{title}</div>', unsafe_allow_html=True)
                st.image(img, use_container_width=True)
                st.markdown(f'<div class="insight-card"><div class="insight-body">{caption}</div></div>',
                            unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — LEARN
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Learn":
    st.markdown('<div class="section-title">Learn</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">The theory, decisions, and intuitions behind every part of this project — in plain language.</div>', unsafe_allow_html=True)

    sections = [
        ("📌 What is customer churn and why does it matter?", True, """
        <strong style='color:#d0cdc6'>Churn</strong> is when a customer stops doing business with a company —
        cancels their subscription, switches to a competitor, or simply disengages.<br><br>
        For a telecom company, every churned customer represents lost recurring revenue. The problem is
        compounded by a well-established business reality: <strong style='color:#d0cdc6'>acquiring a new
        customer costs 5–7× more than retaining an existing one.</strong><br><br>
        If a company could identify at-risk customers <em>before</em> they decide to leave, it could intervene —
        offer a discount, make a call, improve service — and retain them at a fraction of the acquisition cost.
        That is exactly what this model does.
        """),
        ("📊 Why AUC-ROC and Recall — not Accuracy?", False, """
        The dataset has a <strong style='color:#d0cdc6'>class imbalance</strong>: only 26.5% of customers churned.
        A model that predicted "no churn" for everyone would score <strong style='color:#d0cdc6'>73.5% accuracy
        </strong> — and be completely useless.<br><br>
        <strong style='color:#d0cdc6'>AUC-ROC</strong> measures how well the model separates churners from
        non-churners across all decision thresholds. A score of 0.5 = random guessing. 1.0 = perfect.
        Our model scores <strong style='color:#d0cdc6'>0.828</strong> — strong for a real-world problem.<br><br>
        <strong style='color:#d0cdc6'>Recall</strong> measures: of all actual churners, how many did we catch?
        We prioritise recall because <strong style='color:#d0cdc6'>missing a churner is far more costly than a
        false alarm.</strong><br><br>
        <strong style='color:#d0cdc6'>Precision</strong> of 0.51 means roughly half our flags are correct —
        acceptable given we prioritise recall and the cost of false alarms is low.
        """),
        ("🤖 Why Logistic Regression — what about neural networks?", False, """
        We trained and compared four approaches — Logistic Regression, Random Forest, XGBoost (default),
        and XGBoost (tuned). <strong style='color:#d0cdc6'>Logistic Regression won on our two key metrics:
        AUC-ROC (0.828) and Recall (0.743).</strong><br><br>
        The dataset is relatively small (7,032 rows) and the relationship between features and churn is largely
        linear. More complex models don't have enough data complexity to exploit their advantage —
        <strong style='color:#d0cdc6'>more complex does not always mean better.</strong><br><br>
        <strong style='color:#d0cdc6'>Why not neural networks?</strong> They require large amounts of data
        to shine and are black boxes — you can't explain why they made a specific prediction. For a business
        problem where someone needs to act on the output, explainability is not optional. Neural networks
        dominate in image recognition, NLP, and audio — not structured tabular data with thousands of rows.
        """),
        ("⚖️ What is SMOTE and why did we use it?", False, """
        SMOTE stands for <strong style='color:#d0cdc6'>Synthetic Minority Oversampling Technique</strong>.<br><br>
        With 26.5% churners, the model sees far more examples of loyal customers during training.
        Left uncorrected, it learns to be biased toward predicting "no churn" — the safer default
        given the imbalance.<br><br>
        SMOTE fixes this by <strong style='color:#d0cdc6'>synthetically creating new churn examples</strong> —
        not duplicating, but interpolating between real churners in feature space. After SMOTE,
        the training set becomes 50/50.<br><br>
        <strong style='color:#d0cdc6'>Critical rule:</strong> SMOTE is applied only to the training set.
        The test set is never touched — it must reflect the real-world distribution.
        """),
        ("🔍 What is SHAP and how do you read it?", False, """
        SHAP stands for <strong style='color:#d0cdc6'>SHapley Additive exPlanations</strong> — rooted in
        game theory. For any prediction the model makes, SHAP fairly distributes credit across all features,
        like dividing credit among players on a winning team.<br><br>
        <strong style='color:#d0cdc6'>Bar chart:</strong> features ranked by average absolute SHAP value.
        Longer bar = more influence overall.<br><br>
        <strong style='color:#d0cdc6'>Beeswarm:</strong> each dot is one customer. Positive SHAP = pushed
        toward churn. Negative = pushed toward loyalty. Red = high feature value, blue = low.<br><br>
        <strong style='color:#d0cdc6'>Per-customer chart (Predict page):</strong> red bars pushed this specific
        customer toward churn, green pushed toward loyalty. Individual explanation, not a population average.
        """),
        ("🔧 What preprocessing was done and why?", False, """
        <strong style='color:#d0cdc6'>Encoding:</strong> Models can't process raw text. Binary columns
        were label encoded to 1/0. Multi-category columns were one-hot encoded — a separate binary column
        per category — avoiding false ordering implications.<br><br>
        <strong style='color:#d0cdc6'>Scaling:</strong> Tenure ranges 0–72, TotalCharges 18–8,684.
        StandardScaler transforms all numeric columns to mean=0, std=1 — equal footing. The scaler was
        fitted on training data only. No data leakage.<br><br>
        <strong style='color:#d0cdc6'>Train-test split:</strong> 80% training, 20% test, stratified —
        both halves preserve the 26.5%/73.5% churn ratio for honest evaluation.
        """),
        ("💼 What does this mean for the business?", False, """
        <strong style='color:#d0cdc6'>1. Price is the primary lever.</strong> MonthlyCharges dominates
        every other feature. The most impactful retention strategy is targeted pricing intervention
        for high-bill, short-tenure customers — the highest risk segment.<br><br>
        <strong style='color:#d0cdc6'>2. The first year is critical.</strong> Churn is highest among
        0–12 month customers. An onboarding retention programme during the first year has the highest
        ROI of any retention initiative.<br><br>
        <strong style='color:#d0cdc6'>3. Fiber optic customers are high-risk.</strong> They pay more
        and churn more. Targeted loyalty pricing for fiber customers is a direct intervention the model supports.<br><br>
        <strong style='color:#d0cdc6'>4. Long-term contracts are protective.</strong> Incentivising
        month-to-month customers to move to annual contracts — even with a discount — dramatically
        reduces churn exposure. 43% vs 3% churn rate tells the whole story.
        """),
    ]

    for title, expanded, body in sections:
        with st.expander(title, expanded=expanded):
            st.markdown(f'<div class="learn-body">{body}</div>', unsafe_allow_html=True)