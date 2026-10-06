import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                             roc_auc_score, average_precision_score, confusion_matrix)

# Attempt to import xgboost
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("WARNING: xgboost library not found. XGBoost model will be skipped.")

def train_and_evaluate():
    # Paths
    data_path = Path("data/final/chennai_flood_training_dataset.csv")
    models_dir = Path("models")
    reports_dir = Path("reports")
    
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        print(f"ERROR: {data_path} not found. Run combine_features.py first.")
        return

    print("Loading dataset...")
    df = pd.read_csv(data_path)

    # 1. Define Features and Target
    target = 'flood_label'
    
    # Strictly define features to PREVENT TARGET LEAKAGE
    features = [
        'elevation_mean', 
        'slope_mean', 
        'rainfall_24h', 
        'rainfall_72h', 
        'rainfall_7day', 
        'distance_to_water_m', 
        'distance_to_drain_m', 
        'drainage_density', 
        'built_up_percentage', 
        'water_percentage'
    ]

    # Ensure all required features exist in the dataset
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        print(f"ERROR: Missing features in dataset: {missing_features}")
        return

    X = df[features]
    y = df[target]

    # 2. Handle missing/infinite values in features just in case
    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)

    # 3. Train/Test Split
    print("\n--- SPATIAL SPLIT WARNING ---")
    print("Using a standard random 80/20 train-test split.")
    print("Note: Randomly splitting nearby grid cells can produce overoptimistic ")
    print("results because neighbouring cells are spatially correlated (similar elevation,")
    print("similar distance to water). In production, block-based spatial cross-validation ")
    print("should be used to test the model on completely unseen neighborhoods.")
    print("-----------------------------\n")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Calculate class imbalance for XGBoost
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    scale_pos_wt = num_neg / num_pos if num_pos > 0 else 1.0

    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")
    print(f"Training Class balance: {num_pos} Flooded / {num_neg} Safe")

    # 4. Initialize Models
    models = {
        'Logistic_Regression': LogisticRegression(
            max_iter=1000, 
            class_weight='balanced', 
            random_state=42
        ),
        'Random_Forest': RandomForestClassifier(
            n_estimators=100, 
            class_weight='balanced', 
            random_state=42,
            n_jobs=-1
        )
    }

    if XGB_AVAILABLE:
        models['XGBoost'] = xgb.XGBClassifier(
            n_estimators=100,
            scale_pos_weight=scale_pos_wt,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )

    # 5. Train and Evaluate
    metrics_records = []

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # Predict classes and probabilities
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"--- {name} Results ---")
        print(f"Precision: {prec:.3f} | Recall: {rec:.3f} | F1-Score: {f1:.3f}")
        print(f"ROC-AUC:   {roc_auc:.3f} | PR-AUC: {pr_auc:.3f}")
        print(f"Confusion Matrix:\n{cm}")
        
        metrics_records.append({
            'Model': name,
            'Precision': round(prec, 3),
            'Recall': round(rec, 3),
            'F1_Score': round(f1, 3),
            'ROC_AUC': round(roc_auc, 3),
            'PR_AUC': round(pr_auc, 3)
        })
        
        # Save model
        model_path = models_dir / f"{name.lower()}.pkl"
        joblib.dump(model, model_path)
        print(f"Saved model to {model_path}")

    # 6. Save Metrics Report
    metrics_df = pd.DataFrame(metrics_records)
    metrics_path = reports_dir / "model_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nSaved all model evaluation metrics to {metrics_path}")

if __name__ == "__main__":
    train_and_evaluate()