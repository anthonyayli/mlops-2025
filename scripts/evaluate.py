import argparse
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def build_parser():
    p = argparse.ArgumentParser(description="Evaluate trained model")
    p.add_argument("--model", required=True, help="Path to trained model pickle file")
    p.add_argument("--test-features", required=True, help="Path to test features CSV")
    p.add_argument("--test-labels", required=True, help="Path to test labels CSV")
    p.add_argument("--metrics-output", help="Optional path to save metrics as JSON")
    return p

def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def load_data(features_path, labels_path):
    X = pd.read_csv(features_path)
    y = pd.read_csv(labels_path)
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    # Convert to numpy arrays since features are already transformed
    X = X.values
    y = y.values if hasattr(y, 'values') else y
    return X, y

def compute_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }
    return metrics

def print_metrics(metrics):
    print("Model Evaluation Metrics:")
    print("-" * 30)
    for metric_name, value in metrics.items():
        print(f"{metric_name.capitalize()}: {value:.4f}")

def save_metrics(metrics, output_path):
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {output_path}")

def main():
    args = build_parser().parse_args()
    
    model = load_model(args.model)
    X_test, y_test = load_data(args.test_features, args.test_labels)
    
    metrics = compute_metrics(model, X_test, y_test)
    print_metrics(metrics)
    
    if args.metrics_output:
        save_metrics(metrics, args.metrics_output)

if __name__ == "__main__":
    main()