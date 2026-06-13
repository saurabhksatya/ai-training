import streamlit as st
import pandas as pd
import numpy as np

from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.utils import Bunch

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

import matplotlib.pyplot as plt
import seaborn as sns


st.set_page_config(
    page_title="StreamTune",
    page_icon="📊",
    layout="wide",
)

st.title("📊 StreamTune - Interactive Machine Learning Model Tuning")

st.write(
    """
Upload your dataset, compare multiple machine learning models,
and tune hyperparameters interactively.
"""
)

# --------------------------------------------------
# DATASET SELECTION
# --------------------------------------------------

st.sidebar.title("Dataset")

sample_dataset = st.sidebar.selectbox(
    "Select Dataset",
    ["Iris", "Breast Cancer", "Custom"],
)

data = None

if sample_dataset == "Iris":
    data = load_iris()

elif sample_dataset == "Breast Cancer":
    data = load_breast_cancer()

else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV or Excel",
        type=["csv", "xlsx"],
    )

    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            data = pd.read_csv(uploaded_file)

        elif uploaded_file.name.endswith(".xlsx"):
            data = pd.read_excel(uploaded_file)

# --------------------------------------------------
# DATA PREPARATION
# --------------------------------------------------

if data is not None:

    if isinstance(data, Bunch):

        df = pd.DataFrame(
            data=data.data,
            columns=data.feature_names,
        )

        df["target"] = data.target

        if hasattr(data, "target_names"):
            mapping = {
                i: data.target_names[i]
                for i in range(len(data.target_names))
            }

            df["target"] = df["target"].map(mapping)

    else:
        df = data.copy()

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.write(f"Rows: {df.shape[0]}")
    st.write(f"Columns: {df.shape[1]}")

    # --------------------------------------------------
    # HANDLE MISSING VALUES
    # --------------------------------------------------

    if df.isnull().sum().sum() > 0:

        st.warning("Missing values detected. Filling automatically.")

        for col in df.columns:

            if df[col].dtype == "object":
                mode_val = df[col].mode()

                if len(mode_val):
                    df[col] = df[col].fillna(mode_val[0])

            else:
                df[col] = df[col].fillna(df[col].median())

    # --------------------------------------------------
    # TARGET VARIABLE
    # --------------------------------------------------

    st.sidebar.title("Target Variable")

    target_variable = st.sidebar.selectbox(
        "Select Target Column",
        df.columns,
    )

    if target_variable not in df.columns:
        st.error("Invalid target column.")
        st.stop()

    if df[target_variable].nunique() < 2:
        st.error("Target column must contain at least 2 classes.")
        st.stop()

    # --------------------------------------------------
    # HYPERPARAMETERS
    # --------------------------------------------------

    st.sidebar.title("Hyperparameters")

    rf_n_estimators = st.sidebar.slider(
        "Random Forest Trees",
        10,
        500,
        100,
    )

    rf_max_depth = st.sidebar.slider(
        "Random Forest Max Depth",
        1,
        50,
        10,
    )

    knn_neighbors = st.sidebar.slider(
        "KNN Neighbors",
        1,
        20,
        5,
    )

    gb_estimators = st.sidebar.slider(
        "Gradient Boosting Estimators",
        10,
        500,
        100,
    )

    # --------------------------------------------------
    # MODEL SELECTION
    # --------------------------------------------------

    st.sidebar.title("Models")

    selected_models = st.sidebar.multiselect(
        "Choose Models",
        [
            "Random Forest",
            "k-Nearest Neighbors",
            "Logistic Regression",
            "Gradient Boosting",
            "Support Vector Machine",
        ],
    )

    if not selected_models:
        st.warning("Select at least one model.")
        st.stop()

    # --------------------------------------------------
    # FEATURE ENGINEERING
    # --------------------------------------------------

    X = df.drop(columns=[target_variable])
    y = df[target_variable].astype(str)

    # Convert categorical features
    X = pd.get_dummies(X, drop_first=True)

    class_counts = y.value_counts()

    rare_classes = class_counts[class_counts < 2]

    if len(rare_classes) > 0:

        st.warning(
            f"Removing classes with fewer than 2 samples: "
            f"{list(rare_classes.index)}"
        )

        valid_classes = class_counts[class_counts >= 2].index

        mask = y.isin(valid_classes)

        X = X[mask]
        y = y[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --------------------------------------------------
    # MODEL LOOP
    # --------------------------------------------------

    st.header("Model Evaluation")

    for selected_model in selected_models:

        st.markdown("---")
        st.subheader(selected_model)

        # ----------------------------------------
        # MODEL CREATION
        # ----------------------------------------

        if selected_model == "Random Forest":

            model = RandomForestClassifier(
                n_estimators=rf_n_estimators,
                max_depth=rf_max_depth,
                random_state=42,
            )

        elif selected_model == "k-Nearest Neighbors":

            model = KNeighborsClassifier(
                n_neighbors=knn_neighbors,
            )

        elif selected_model == "Logistic Regression":

            model = LogisticRegression(
                max_iter=1000,
            )

        elif selected_model == "Gradient Boosting":

            model = GradientBoostingClassifier(
                n_estimators=gb_estimators,
                random_state=42,
            )

        elif selected_model == "Support Vector Machine":

            model = SVC(
                kernel="linear",
                C=1.0,
                random_state=42,
            )

        # ----------------------------------------
        # SCALED OR UNSCALED
        # ----------------------------------------

        use_scaled = selected_model in [
            "k-Nearest Neighbors",
            "Logistic Regression",
            "Support Vector Machine",
        ]

        if use_scaled:

            model.fit(X_train_scaled, y_train)

            y_pred = model.predict(X_test_scaled)

            cv_score = cross_val_score(
                model,
                scaler.fit_transform(X),
                y,
                cv=5,
            ).mean()

        else:

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            cv_score = cross_val_score(
                model,
                X,
                y,
                cv=5,
            ).mean()

        # ----------------------------------------
        # METRICS
        # ----------------------------------------

        accuracy = accuracy_score(y_test, y_pred)

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Accuracy",
                f"{accuracy:.4f}",
            )

        with col2:
            st.metric(
                "Cross Validation Score",
                f"{cv_score:.4f}",
            )

        # ----------------------------------------
        # CONFUSION MATRIX
        # ----------------------------------------

        st.write("### Confusion Matrix")

        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(6, 4))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
        )

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        st.pyplot(fig)

        plt.close(fig)

        # ----------------------------------------
        # CLASSIFICATION REPORT
        # ----------------------------------------

        st.write("### Classification Report")

        report = classification_report(
            y_test,
            y_pred,
            output_dict=True,
        )

        report_df = (
            pd.DataFrame(report)
            .transpose()
            .round(4)
        )

        st.dataframe(
            report_df,
            use_container_width=True,
        )

# --------------------------------------------------
# FEEDBACK FORM
# --------------------------------------------------

st.markdown("---")

feedback_expander = st.expander(
    "Feedback Form",
    expanded=False,
)

with feedback_expander:

    with st.form("feedback_form"):

        st.subheader("We'd Love Your Feedback")

        feedback_text = st.text_area(
            "Feedback",
            height=150,
        )

        submitted = st.form_submit_button(
            "Submit Feedback"
        )

        if submitted:

            with open("feedback.txt", "a", encoding="utf-8") as f:
                f.write(feedback_text + "\n")

            st.success(
                "Feedback submitted successfully. Thank you!"
            )