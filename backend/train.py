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

import pandas as pd


def train_models(
    df: pd.DataFrame,
    target_variable: str,
    selected_models: list,
    rf_n_estimators: int = 100,
    rf_max_depth: int = 10,
    knn_neighbors: int = 5,
    gb_estimators: int = 100,
):
    # ----------------------------------------
    # Missing Values
    # ----------------------------------------

    if df.isnull().sum().sum() > 0:

        for col in df.columns:

            if df[col].dtype == "object":

                mode_val = df[col].mode()

                if len(mode_val):
                    df[col] = df[col].fillna(mode_val[0])

            else:
                df[col] = df[col].fillna(df[col].median())

    X = df.drop(columns=[target_variable])
    y = df[target_variable].astype(str)

    # One-hot encode categorical features
    X = pd.get_dummies(X, drop_first=True)

    # Remove rare classes
    class_counts = y.value_counts()

    valid_classes = class_counts[class_counts >= 2].index

    mask = y.isin(valid_classes)

    X = X[mask]
    y = y[mask]

    if len(y.unique()) < 2:
        raise ValueError(
            "Target must contain at least 2 classes after filtering."
        )


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


    results = {}

    for selected_model in selected_models:
        if selected_model == "Random Forest":

            model = RandomForestClassifier(
                n_estimators=rf_n_estimators,
                max_depth=rf_max_depth,
                random_state=42,
            )

        elif selected_model == "k-Nearest Neighbors":

            model = KNeighborsClassifier(
                n_neighbors=knn_neighbors
            )

        elif selected_model == "Logistic Regression":

            model = LogisticRegression(
                max_iter=1000
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

        else:
            continue

        scaled_models = [
            "k-Nearest Neighbors",
            "Logistic Regression",
            "Support Vector Machine",
        ]

        if selected_model in scaled_models:

            model.fit(
                X_train_scaled,
                y_train,
            )

            y_pred = model.predict(
                X_test_scaled
            )

            cv_score = cross_val_score(
                model,
                scaler.fit_transform(X),
                y,
                cv=5,
            ).mean()

        else:

            model.fit(
                X_train,
                y_train,
            )

            y_pred = model.predict(
                X_test
            )

            cv_score = cross_val_score(
                model,
                X,
                y,
                cv=5,
            ).mean()


        accuracy = accuracy_score(
            y_test,
            y_pred,
        )

        cm = confusion_matrix(
            y_test,
            y_pred,
        )

        report = classification_report(
            y_test,
            y_pred,
            output_dict=True,
        )

        results[selected_model] = {
            "accuracy": float(accuracy),
            "cv_score": float(cv_score),
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        }

    return {
        "dataset_rows": int(len(df)),
        "dataset_columns": int(df.shape[1]),
        "target_variable": target_variable,
        "models": results,
    }