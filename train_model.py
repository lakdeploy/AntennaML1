import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

def load_and_clean(filepath='zmmwave_6g_continuous_dataset.csv'):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ Dataset not found at {filepath}")
    
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    df.dropna(inplace=True)
    return df

def train_and_evaluate_model():
    df = load_and_clean()

    X = df.drop('spectral_efficiency', axis=1)
    y = df['spectral_efficiency']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        ))
    ])

    print("🚀 Training model...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"✅ Test MSE: {mse:.4f}")
    print(f"✅ Test R²: {r2:.4f}")

    # Save the model
    joblib.dump(pipeline, 'model/rf_model.pkl')
    print("💾 Model saved to model/rf_model.pkl")

    # Plot: Actual vs Predicted
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, edgecolor='k', alpha=0.7)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel("Actual Spectral Efficiency")
    plt.ylabel("Predicted Spectral Efficiency")
    plt.title("Actual vs Predicted Spectral Efficiency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("assets/prediction_plot.png")
    print("📊 Plot saved to assets/prediction_plot.png")
    plt.show()

if __name__ == "__main__":
    train_and_evaluate_model()
