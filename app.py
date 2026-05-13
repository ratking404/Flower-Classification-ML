import joblib
from sklearn.tree import export_text

from src.config import MODEL_PATH, REPORTS_DIR, FEATURE_COLUMNS
from src.data_loader import load_dataset, split_features_and_target
from src.model_trainer import create_train_test_data, train_model
from src.evaluator import evaluate_model
from src.visualizer import (
    create_species_chart,
    create_feature_importance_chart,
    create_decision_tree_chart,
)
from src.predictor import predict_flower


def main():
    print("Starting Benedict Flower Classification Project...")

    data = load_dataset()
    print(f"Dataset loaded successfully. Rows: {len(data)}")

    X, y = split_features_and_target(data)
    X_train, X_test, y_train, y_test = create_train_test_data(X, y)

    print("Training the Decision Tree model...")
    model = train_model(X_train, y_train)

    predictions, accuracy, report, matrix = evaluate_model(model, X_test, y_test)

    print("\nMODEL RESULTS")
    print("-------------")
    print(f"Accuracy: {accuracy:.2%}")
    print("\nClassification report:")
    print(report)

    print("Saving model, reports, and charts...")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    (REPORTS_DIR / "classification_report.txt").write_text(report)
    (REPORTS_DIR / "metrics_summary.txt").write_text(
        f"Accuracy: {accuracy:.2%}\nTraining rows: {len(X_train)}\nTesting rows: {len(X_test)}\n"
    )
    (REPORTS_DIR / "decision_rules.txt").write_text(
        export_text(model, feature_names=FEATURE_COLUMNS)
    )

    create_species_chart(data)
    create_feature_importance_chart(model)
    create_decision_tree_chart(model)

    example_prediction = predict_flower(model, 5.1, 3.5, 1.4, 0.2)
    print(f"\nExample prediction for measurements 5.1, 3.5, 1.4, 0.2: {example_prediction}")

    print("\nDone. Check the outputs folder for charts and reports.")


if __name__ == "__main__":
    main()
