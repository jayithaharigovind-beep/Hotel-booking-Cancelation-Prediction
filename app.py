import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Hotel Booking Cancellation Predictor",
    page_icon="🏨",
    layout="wide"
)

# ============================================================
# LOAD MODEL, DATA AND RESULTS
# ============================================================

model = joblib.load("hotel_booking_model.pkl")
data = pd.read_csv("hotel_bookings.csv")

comparison_file = Path("model_comparison.csv")
best_model_file = Path("best_model_name.txt")

if comparison_file.exists():
    comparison_df = pd.read_csv(comparison_file)
else:
    comparison_df = pd.DataFrame()

if best_model_file.exists():
    best_model_name = best_model_file.read_text(
        encoding="utf-8"
    ).strip()
else:
    best_model_name = "XGBoost"

# ============================================================
# HEADER
# ============================================================

st.title("🏨 Hotel Booking Cancellation Predictor")
st.caption(
    "Machine Learning Decision-Support Dashboard | "
    "Hotel Booking Cancellation Prediction"
)

# ============================================================
# TABS
# ============================================================

overview_tab, insights_tab, models_tab, prediction_tab, business_tab = st.tabs(
    [
        "🏠 Overview",
        "📊 Data Insights",
        "🤖 Model Comparison",
        "🔮 Prediction",
        "💼 Business Insights"
    ]
)

# ============================================================
# 1. OVERVIEW
# ============================================================

with overview_tab:

    st.header("Project Overview")

    st.write(
        """
        This project develops a machine learning system to predict whether
        a hotel booking is likely to be cancelled. The objective is to use
        historical booking information to identify cancellation risk and
        support proactive hotel management decisions.
        """
    )

    st.subheader("Business Problem")

    st.write(
        """
        Hotel booking cancellations can create uncertainty in occupancy
        planning, revenue management and resource allocation. Predicting
        cancellation risk in advance can help hotels take appropriate
        actions before the expected arrival date.
        """
    )

    st.subheader("Machine Learning Objective")

    st.write(
        """
        The target variable is `is_canceled`, where:
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        st.info("0 = Booking is not cancelled")

    with col2:
        st.warning("1 = Booking is cancelled")

    st.subheader("Dataset Summary")

    clean_data = data.drop_duplicates()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Original Records", f"{len(data):,}")

    with c2:
        st.metric("Records After Cleaning", f"{len(clean_data):,}")

    with c3:
        st.metric("Original Features", "32")

    with c4:
        st.metric("Models Compared", "9")

    st.subheader("Methodology")

    st.write(
        """
        1. Data loading and inspection  
        2. Duplicate removal  
        3. Removal of post-outcome information and high-missing-value fields  
        4. Feature preprocessing and encoding  
        5. Stratified 80:20 train-test split  
        6. Training of nine classification models  
        7. Evaluation using Accuracy, Precision, Recall and F1 Score  
        8. Selection of the final model using F1 Score  
        9. Deployment through an interactive Streamlit dashboard
        """
    )

    st.subheader("Models Evaluated")

    st.write(
        """
        Logistic Regression • Decision Tree • Naive Bayes • KNN • SVM •
        Random Forest • AdaBoost • XGBoost • Artificial Neural Network
        """
    )

# ============================================================
# 2. DATA INSIGHTS
# ============================================================

with insights_tab:

    st.header("📊 Data Insights")

    st.write(
        "Explore historical cancellation patterns in the hotel booking dataset."
    )

    clean_data = data.drop_duplicates()

    # Hotel type
    hotel_cancel = (
        clean_data.groupby("hotel")["is_canceled"]
        .mean()
        .mul(100)
        .reset_index()
    )

    st.subheader("Cancellation Rate by Hotel Type")
    st.bar_chart(hotel_cancel.set_index("hotel"))

    # Market segment
    segment_cancel = (
        clean_data.groupby("market_segment")["is_canceled"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )

    st.subheader("Cancellation Rate by Market Segment")
    st.bar_chart(segment_cancel)

    # Deposit type
    deposit_cancel = (
        clean_data.groupby("deposit_type")["is_canceled"]
        .mean()
        .mul(100)
    )

    st.subheader("Cancellation Rate by Deposit Type")
    st.bar_chart(deposit_cancel)

    # Lead time
    clean_data["lead_time_group"] = pd.cut(
        clean_data["lead_time"],
        bins=[-1, 30, 90, 180, 365, 1000],
        labels=[
            "0–30 days",
            "31–90 days",
            "91–180 days",
            "181–365 days",
            "365+ days"
        ]
    )

    lead_cancel = (
        clean_data.groupby(
            "lead_time_group",
            observed=False
        )["is_canceled"]
        .mean()
        .mul(100)
    )

    st.subheader("Cancellation Rate by Lead Time")
    st.line_chart(lead_cancel)

# ============================================================
# 3. MODEL COMPARISON
# ============================================================

with models_tab:

    st.header("🤖 Model Performance & Comparison")

    st.write(
        """
        Nine applicable classification models were evaluated using the
        same held-out 80:20 stratified train-test split. F1 Score was used
        as the primary criterion for selecting the final model.
        """
    )

    if not comparison_df.empty:

        display_df = comparison_df.copy()

        for col in ["Accuracy", "Precision", "Recall", "F1 Score"]:
            display_df[col] = (
                display_df[col] * 100
            ).round(2).astype(str) + "%"

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("F1 Score Comparison")

        chart_df = comparison_df.set_index("Model")[["F1 Score"]]
        st.bar_chart(chart_df)

        st.subheader("Selected Model")

        selected_row = comparison_df[
            comparison_df["Model"] == best_model_name
        ]

        if not selected_row.empty:

            row = selected_row.iloc[0]

            st.success(
                f"Final model selected: {best_model_name}"
            )

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.metric("Accuracy", f"{row['Accuracy']:.2%}")

            with c2:
                st.metric("Precision", f"{row['Precision']:.2%}")

            with c3:
                st.metric("Recall", f"{row['Recall']:.2%}")

            with c4:
                st.metric("F1 Score", f"{row['F1 Score']:.2%}")

            st.write(
                f"{best_model_name} was selected because it achieved "
                "the highest F1 Score on the held-out test dataset. "
                "F1 Score provides a balance between precision and recall."
            )

    else:
        st.warning(
            "Model comparison results were not found. "
            "Run train_model.py first."
        )

# ============================================================
# 4. PREDICTION
# ============================================================

with prediction_tab:

    st.header("🔮 Predict Cancellation Risk")

    st.write(
        """
        Enter the booking characteristics below and use the trained
        machine learning model to estimate cancellation risk.
        """
    )

    st.info(f"Currently deployed model: **{best_model_name}**")

    st.subheader("Booking Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        hotel = st.selectbox(
            "Hotel Type",
            ["Resort Hotel", "City Hotel"]
        )

        lead_time = st.number_input(
            "Lead Time (days)",
            min_value=0,
            max_value=1000,
            value=50
        )

        arrival_date_year = st.selectbox(
            "Arrival Year",
            [2015, 2016, 2017]
        )

        arrival_date_month = st.selectbox(
            "Arrival Month",
            [
                "January", "February", "March", "April",
                "May", "June", "July", "August",
                "September", "October", "November", "December"
            ]
        )

        arrival_date_week_number = st.number_input(
            "Arrival Week Number",
            min_value=1,
            max_value=53,
            value=25
        )

        arrival_date_day_of_month = st.number_input(
            "Arrival Day",
            min_value=1,
            max_value=31,
            value=15
        )

        stays_in_weekend_nights = st.number_input(
            "Weekend Nights",
            min_value=0,
            max_value=20,
            value=1
        )

        stays_in_week_nights = st.number_input(
            "Week Nights",
            min_value=0,
            max_value=50,
            value=2
        )

        adults = st.number_input(
            "Adults",
            min_value=0,
            max_value=10,
            value=2
        )

        children = st.number_input(
            "Children",
            min_value=0.0,
            max_value=10.0,
            value=0.0
        )

    with col2:

        babies = st.number_input(
            "Babies",
            min_value=0,
            max_value=10,
            value=0
        )

        meal = st.selectbox(
            "Meal",
            ["BB", "HB", "FB", "SC", "Undefined"]
        )

        market_segment = st.selectbox(
            "Market Segment",
            [
                "Online TA", "Offline TA/TO", "Groups", "Direct",
                "Corporate", "Complementary", "Aviation", "Undefined"
            ]
        )

        distribution_channel = st.selectbox(
            "Distribution Channel",
            ["TA/TO", "Direct", "Corporate", "GDS", "Undefined"]
        )

        is_repeated_guest = st.selectbox(
            "Repeated Guest?",
            [0, 1]
        )

        previous_cancellations = st.number_input(
            "Previous Cancellations",
            min_value=0,
            max_value=20,
            value=0
        )

        previous_bookings_not_canceled = st.number_input(
            "Previous Non-Cancelled Bookings",
            min_value=0,
            max_value=100,
            value=0
        )

        reserved_room_type = st.text_input(
            "Reserved Room Type",
            value="A"
        )

        assigned_room_type = st.text_input(
            "Assigned Room Type",
            value="A"
        )

        booking_changes = st.number_input(
            "Booking Changes",
            min_value=0,
            max_value=20,
            value=0
        )

        deposit_type = st.selectbox(
            "Deposit Type",
            ["No Deposit", "Non Refund", "Refundable"]
        )

    with col3:

        days_in_waiting_list = st.number_input(
            "Days in Waiting List",
            min_value=0,
            max_value=400,
            value=0
        )

        customer_type = st.selectbox(
            "Customer Type",
            ["Transient", "Contract", "Transient-Party", "Group"]
        )

        adr = st.number_input(
            "Average Daily Rate",
            min_value=0.0,
            max_value=1000.0,
            value=100.0
        )

        required_car_parking_spaces = st.number_input(
            "Required Parking Spaces",
            min_value=0,
            max_value=10,
            value=0
        )

        total_of_special_requests = st.number_input(
            "Special Requests",
            min_value=0,
            max_value=10,
            value=0
        )

        st.markdown("**Additional dataset fields**")
        country = st.text_input(
            "Country Code",
            value="PRT"
        )

        agent = st.text_input(
            "Agent",
            value=""
        )

    st.divider()

    if st.button(
        "🔮 Predict Cancellation Risk",
        type="primary",
        use_container_width=True
    ):

        agent_value = None if agent.strip() == "" else agent

        input_data = pd.DataFrame({
            "hotel": [hotel],
            "lead_time": [lead_time],
            "arrival_date_year": [arrival_date_year],
            "arrival_date_month": [arrival_date_month],
            "arrival_date_week_number": [arrival_date_week_number],
            "arrival_date_day_of_month": [arrival_date_day_of_month],
            "stays_in_weekend_nights": [stays_in_weekend_nights],
            "stays_in_week_nights": [stays_in_week_nights],
            "adults": [adults],
            "children": [children],
            "babies": [babies],
            "meal": [meal],
            "country": [country],
            "market_segment": [market_segment],
            "distribution_channel": [distribution_channel],
            "is_repeated_guest": [is_repeated_guest],
            "previous_cancellations": [previous_cancellations],
            "previous_bookings_not_canceled": [
                previous_bookings_not_canceled
            ],
            "reserved_room_type": [reserved_room_type],
            "assigned_room_type": [assigned_room_type],
            "booking_changes": [booking_changes],
            "deposit_type": [deposit_type],
            "agent": [agent_value],
            "days_in_waiting_list": [days_in_waiting_list],
            "customer_type": [customer_type],
            "adr": [adr],
            "required_car_parking_spaces": [
                required_car_parking_spaces
            ],
            "total_of_special_requests": [
                total_of_special_requests
            ]
        })

        prediction = model.predict(input_data)[0]

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_data)[0][1]
        else:
            probability = None

        st.subheader("Prediction Result")

        if prediction == 1:

            if probability is not None:
                st.error(
                    f"⚠️ High Cancellation Risk\n\n"
                    f"Estimated cancellation probability: "
                    f"{probability:.1%}"
                )
            else:
                st.error("⚠️ High Cancellation Risk")

        else:

            if probability is not None:
                st.success(
                    f"✅ Low Cancellation Risk\n\n"
                    f"Estimated cancellation probability: "
                    f"{probability:.1%}"
                )
            else:
                st.success("✅ Low Cancellation Risk")

# ============================================================
# 5. BUSINESS INSIGHTS
# ============================================================

with business_tab:

    st.header("💼 Business Insights")

    st.subheader("Why Cancellation Prediction Matters")

    st.write(
        """
        A cancellation prediction system can support hotels in identifying
        bookings that may require proactive attention. This can contribute
        to better occupancy planning, customer communication and revenue
        management.
        """
    )

    st.subheader("Potential Managerial Actions")

    st.markdown(
        """
        - Send targeted booking confirmation or reminder messages.
        - Prioritize proactive communication for higher-risk bookings.
        - Review deposit and booking-policy structures.
        - Use cancellation-risk information alongside occupancy forecasts.
        - Support revenue-management and capacity-planning decisions.
        """
    )

    st.subheader("Model Performance")

    if not comparison_df.empty:

        selected_row = comparison_df[
            comparison_df["Model"] == best_model_name
        ]

        if not selected_row.empty:

            row = selected_row.iloc[0]

            st.write(
                f"The selected **{best_model_name}** achieved an "
                f"accuracy of **{row['Accuracy']:.2%}**, precision of "
                f"**{row['Precision']:.2%}**, recall of "
                f"**{row['Recall']:.2%}**, and F1 Score of "
                f"**{row['F1 Score']:.2%}** on the held-out test dataset."
            )

    st.subheader("Important Interpretation")

    st.write(
        """
        Model predictions should be treated as decision-support information.
        They should be considered together with operational context and
        managerial judgement rather than being used as an automatic
        replacement for human decision-making.
        """
    )

    st.subheader("Project Limitation")

    st.write(
        """
        The model is evaluated using historical booking data and a held-out
        test set. Its performance may differ when applied to new hotels,
        different customer populations or booking environments.
        """
    )
