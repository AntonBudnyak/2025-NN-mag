import os
import logging
import warnings
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import kagglehub

from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    classification_report,
    confusion_matrix,
    RocCurveDisplay,
)

from gama import GamaClassifier


warnings.simplefilter("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


PALETTE = {
    "primary": "#2a9d8f",
    "secondary": "#e9c46a",
    "accent": "#e76f51",
}


def ensure_reports_dir() -> str:
    """
    Create reports directory next to the project root if it does not exist.
    :return: Path to the reports directory.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    reports_dir = os.path.join(project_root, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    return reports_dir


def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load the Pima Indians Diabetes dataset from a CSV file.
    URL: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
    :param csv_path: Path to the CSV file.
    :return: DataFrame with the dataset.
    """
    if os.path.isdir(csv_path):
        csv_files = [f for f in os.listdir(csv_path) if f.endswith(".csv")]
        if csv_files:
            csv_path = os.path.join(csv_path, csv_files[0])
        else:
            raise FileNotFoundError(
                f"No CSV files found in directory {csv_path}. " f"Check the structure of the downloaded dataset."
            )

    if not os.path.exists(csv_path):
        logger.error(
            f"File {csv_path} not found. " f"Download 'diabetes.csv' from Kaggle and place it next to the script."
        )
        raise FileNotFoundError(
            f"File {csv_path} not found. " f"Download 'diabetes.csv' from Kaggle and place it next to the script."
        )
    df = pd.read_csv(csv_path)
    logger.info(f"Data loaded from {csv_path}")
    return df


def plot_target_distribution(df: pd.DataFrame, target_col: str, output_dir: str) -> None:
    """
    Plot and save target distribution.
    :param df: DataFrame with the dataset.
    :param target_col: Name of the target variable column.
    :param output_dir: Directory to save the plot.
    :return: None
    """
    if target_col not in df.columns:
        logger.warning("Target column missing, skip target distribution plot")
        return
    counts = df[target_col].value_counts(dropna=False).sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind="bar", ax=ax, color=PALETTE["primary"])
    ax.set_title("Target distribution")
    ax.set_xlabel(target_col)
    ax.set_ylabel("Count")
    plt.tight_layout()
    path = os.path.join(output_dir, "target_distribution.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    logger.info(f"Target distribution saved to {path}")


def plot_correlation_matrix(df: pd.DataFrame, output_dir: str) -> None:
    """
    Plot and save correlation matrix for numeric columns.
    :param df: DataFrame with the dataset.
    :param output_dir: Directory to save the plot.
    :return: None
    """
    corr = df.corr(numeric_only=True)
    if corr.empty:
        logger.warning("Correlation matrix is empty; skipping plot")
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(corr, cmap="viridis", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Correlation matrix")
    plt.tight_layout()
    path = os.path.join(output_dir, "correlation_matrix.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    logger.info(f"Correlation matrix saved to {path}")


def _get_final_estimator(model) -> object:
    """
    Return final estimator from pipeline if present.
    :param model: sklearn pipeline or estimator.
    :return: Final estimator object.
    """
    if hasattr(model, "steps") and model.steps:
        return model.steps[-1][1]
    if hasattr(model, "named_steps") and model.named_steps:
        last_key = list(model.named_steps.keys())[-1]
        return model.named_steps[last_key]
    return model


def plot_feature_importance(
    model: object,
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: str,
    scoring: str = "roc_auc",
) -> None:
    """
    Plot feature importance using estimator attributes or permutation importance.
    :param model: Trained model (sklearn pipeline or estimator).
    :param X: Training features DataFrame.
    :param y: Training target Series.
    :param output_dir: Directory to save the plot.
    :param scoring: Scoring metric for permutation importance.
    :return: None
    """
    estimator = _get_final_estimator(model)

    importances: Optional[np.ndarray] = None
    if hasattr(estimator, "feature_importances_"):
        importances = np.asarray(estimator.feature_importances_)
    elif hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_)
        importances = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
    else:
        try:
            result = permutation_importance(
                model,
                X,
                y,
                scoring=scoring,
                n_repeats=5,
                random_state=42,
                n_jobs=-1,
            )
            importances = result.importances_mean
        except Exception as ex:
            logger.warning(f"Could not compute permutation importances: {ex}")
            return

    if importances is None or len(importances) != X.shape[1]:
        logger.warning("Feature importances length mismatch; skipping plot")
        return

    feature_names = list(X.columns)
    order = np.argsort(importances)[::-1]
    feature_names = [feature_names[i] for i in order]
    importances = importances[order]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(feature_names[::-1], importances[::-1], color=PALETTE["secondary"])
    ax.set_title("Feature importance (top to bottom)")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    path = os.path.join(output_dir, "feature_importance.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    logger.info(f"Feature importance saved to {path}")


def train_test_split_data(
    df: pd.DataFrame, target_col: str = "Outcome"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the DataFrame into X, y and train/test sets.
    :param df: DataFrame with the dataset.
    :param target_col: Name of the target variable column.
    :return: Tuple of train/test splits: X_train, X_test, y_train, y_test
    """
    if target_col not in df.columns:
        logger.error(
            f"The DataFrame does not contain the column '{target_col}'. " f"Check the target variable name in the CSV."
        )
        raise ValueError(
            f"The DataFrame does not contain the column '{target_col}'. " f"Check the target variable name in the CSV."
        )

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )
    logger.info(f"Train/Test split completed - Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def run_gama_automl(X_train: pd.DataFrame, y_train: pd.Series) -> GamaClassifier:
    """
    Run GAMA AutoML.
    :param X_train: Training features.
    :param y_train: Training target.
    :return: Trained GamaClassifier model.
    """
    num_cpus = os.cpu_count() or 1
    num_workers = max(1, num_cpus // 2)
    logger.info(f"Number of CPU cores available: {num_cpus}")
    logger.info("Initializing GAMA AutoML...")
    gama = GamaClassifier(
        max_total_time=3600,
        max_eval_time=120,
        scoring="roc_auc",
        n_jobs=num_workers,
        random_state=42,
    )

    logger.info("Running GAMA AutoML...")
    gama.fit(X_train, y_train)
    logger.info("GAMA AutoML search completed.")

    return gama


def evaluate_model(model: GamaClassifier, X_test: pd.DataFrame, y_test: pd.Series, reports_dir: str) -> None:
    """
    Evaluate the model on the test set and provide a simple ROC curve visualization.
    :param model: Trained GamaClassifier model.
    :param X_test: Test features.
    :param y_test: Test target.
    :param reports_dir: Directory to save evaluation reports.
    :return: None
    """
    logger.info("Evaluating model on test set...")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    logger.info(f"ROC-AUC: {roc_auc:.4f}")
    logger.info(f"F1-score: {f1:.4f}")
    logger.info(f"Classification report:\n{classification_report(y_test, y_pred)}")
    logger.info(f"Confusion matrix:\n{cm}")

    RocCurveDisplay.from_predictions(y_test, y_proba)
    for line in plt.gca().lines:
        line.set_color(PALETTE["primary"])
    plt.title("ROC curve – GAMA AutoML (Pima Diabetes)")
    plt.tight_layout()

    roc_path = os.path.join(reports_dir, "roc_curve.png")
    plt.savefig(roc_path, dpi=300)
    logger.info(f"ROC curve saved to {roc_path}")
    plt.close()


def save_model(model: GamaClassifier, path: str = "gama_pima_diabetes.joblib") -> None:
    """
    Save the trained GAMA model to a file.
    :param model: Trained GamaClassifier model.
    :param path: Path to save the model file.
    :return: None
    """
    if hasattr(model, "model"):
        joblib.dump(model.model, path)
        logger.info(f"Sklearn pipeline saved: {path}")
    else:
        logger.error("GAMA model does not contain a .model attribute")
        raise ValueError("GAMA model does not contain a .model attribute")


def main():
    """
    Main function to run the GAMA AutoML on the Pima Indians Diabetes dataset.
    1. Load data
    2. Train/Test split
    3. Run GAMA AutoML
    4. Evaluation
    5. Save model
    """
    reports_dir = ensure_reports_dir()

    path = kagglehub.dataset_download("uciml/pima-indians-diabetes-database")

    # 1. Load data
    df = load_data(path)
    logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns.")
    logger.info(f"First rows:\n{df.head()}")

    plot_target_distribution(df, target_col="Outcome", output_dir=reports_dir)
    plot_correlation_matrix(df, output_dir=reports_dir)

    # 2. Train/Test split
    X_train, X_test, y_train, y_test = train_test_split_data(df, target_col="Outcome")

    # 3. Run GAMA AutoML
    gama_model = run_gama_automl(X_train, y_train)

    # 4. Evaluation
    evaluate_model(gama_model, X_test, y_test, reports_dir)
    plot_feature_importance(gama_model.model, X_train, y_train, output_dir=reports_dir)
    plot_model_comparison(gama_model, output_dir=reports_dir)

    # 5. Save model
    save_model(gama_model, "gama_pima_diabetes.joblib")


if __name__ == "__main__":
    main()
