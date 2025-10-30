import argparse
import os
import json
import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
def evaluate(y_true, y_pred):
 mae = mean_absolute_error(y_true, y_pred)
 rmse = np.sqrt(mean_squared_error(y_true, y_pred))
 r2 = r2_score(y_true, y_pred)
 return {'MAE': float(mae), 'RMSE': float(rmse), 'R2': float(r2)}




def main(data_csv, out_dir):
 os.makedirs(out_dir, exist_ok=True)
 df = pd.read_csv(data_csv)
 X = df.drop(columns=['Delivery_Time'])
 y = df['Delivery_Time']
 # Fill any missing numeric values with column medians
 X = X.fillna(X.median())


 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


 models = {
 'LinearRegression': LinearRegression(),
 'RandomForest': RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
 'GradientBoosting': GradientBoostingRegressor(n_estimators=200, random_state=42)
 }


 best_model = None
 best_rmse = float('inf')
 best_name = None
 results = {}


 mlflow.set_experiment('Amazon_Delivery_Time')


 for name, model in models.items():
  with mlflow.start_run(run_name=name):
   model.fit(X_train, y_train)
   preds = model.predict(X_test)
   metrics = evaluate(y_test, preds)
   mlflow.log_metrics(metrics)
   mlflow.sklearn.log_model(model, artifact_path='model')
   results[name] = metrics
   if metrics['RMSE'] < best_rmse:
    best_rmse = metrics['RMSE']
    best_model = model
    best_name = name


 # Save best model
 model_path = os.path.join(out_dir, 'best_model.pkl')
 joblib.dump(best_model, model_path)
 metrics_path = os.path.join(out_dir, 'metrics.json')
 with open(metrics_path, 'w') as f:
  json.dump({'best_model': best_name, 'results': results}, f, indent=2)
 print('Saved best model:', model_path)
 print('Metrics written to', metrics_path)


if __name__ == '__main__':
 import argparse
 parser = argparse.ArgumentParser()
 parser.add_argument('--data_csv', required=True)
 parser.add_argument('--out_dir', default='models')
 args = parser.parse_args()
 main(args.data_csv, args.out_dir)
