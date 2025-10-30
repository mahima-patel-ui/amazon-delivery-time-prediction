# Amazon Delivery Time Prediction


Predict delivery times for Amazon orders using regression models.


## Files
- `data_prep.py` — cleans data and produces `data/processed_amazon_delivery.csv` from `amazon_delivery.csv` (put dataset in project root).
- `train_model.py` — trains models, compares metrics (MAE, RMSE, R2), saves best model to `models/best_model.pkl` and logs runs to MLflow.
- `app.py` — Streamlit app for inference.
- `utils.py` — helper functions.
- `requirements.txt` — environment dependencies.


## How to run (local machine)
1. Put the provided `amazon_delivery.csv` in the project root (same folder as these scripts).
2. Create a virtual environment and install requirements:


```bash
python -m venv venv
source venv/bin/activate # mac/linux
venv\Scripts\activate # windows
pip install -r requirements.txt