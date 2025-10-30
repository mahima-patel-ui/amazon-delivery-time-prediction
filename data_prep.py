import argparse
import os
import pandas as pd
import numpy as np
from dateutil import parser
from utils import haversine_distance, SimpleEncoder

def preprocess(input_csv: str) -> pd.DataFrame:
    """Reads raw CSV, cleans, engineers features, and returns processed DataFrame."""
    df = pd.read_csv(input_csv)

    expected = [
        'Order_ID','Agent_Age','Agent_Rating','Store_Latitude','Store_Longitude',
        'Drop_Latitude','Drop_Longitude','Order_Date','Order_Time','Pickup_Time',
        'Weather','Traffic','Vehicle','Area','Delivery_Time','Category'
    ]
    # Keep only those available in dataset
    cols = [c for c in expected if c in df.columns]
    df = df[cols].copy()

    # Combine date and time
    if 'Order_Date' in df.columns and 'Order_Time' in df.columns:
        df['Order_DateTime'] = pd.to_datetime(
            df['Order_Date'].astype(str).str.strip() + ' ' + df['Order_Time'].astype(str).str.strip(),
            errors='coerce'
        )
    else:
        df['Order_DateTime'] = pd.to_datetime(df.get('Order_Date', None), errors='coerce')

    # Pickup lag
    if 'Pickup_Time' in df.columns:
        df['Pickup_DateTime'] = pd.to_datetime(df['Pickup_Time'], errors='coerce')
        mask = df['Pickup_DateTime'].isna()
        if mask.any() and 'Order_DateTime' in df:
            try:
                df.loc[mask, 'Pickup_DateTime'] = pd.to_datetime(
                    df.loc[mask, 'Order_DateTime'].dt.date.astype(str) + ' ' + df.loc[mask, 'Pickup_Time'].astype(str),
                    errors='coerce'
                )
            except Exception:
                pass
        df['Pickup_Lag_min'] = (df['Pickup_DateTime'] - df['Order_DateTime']).dt.total_seconds() / 60.0
    else:
        df['Pickup_Lag_min'] = np.nan

    # Distance
    if set(['Store_Latitude','Store_Longitude','Drop_Latitude','Drop_Longitude']).issubset(df.columns):
        df['Distance_km'] = haversine_distance(
            df['Store_Latitude'].astype(float), df['Store_Longitude'].astype(float),
            df['Drop_Latitude'].astype(float), df['Drop_Longitude'].astype(float)
        )
    else:
        df['Distance_km'] = np.nan

    # Time features
    df['Hour'] = df['Order_DateTime'].dt.hour
    df['DayOfWeek'] = df['Order_DateTime'].dt.dayofweek

    # Fill numeric missing
    for num in ['Agent_Age','Agent_Rating','Distance_km','Pickup_Lag_min']:
        if num in df.columns:
            df[num] = pd.to_numeric(df[num], errors='coerce')
            df[num] = df[num].fillna(df[num].median())

    # Encode categorical
    cat_cols = [c for c in ['Weather','Traffic','Vehicle','Area','Category'] if c in df.columns]
    encoder = SimpleEncoder()
    if cat_cols:
        cat_df = encoder.fit_transform(df, cat_cols)
        df = pd.concat([df, cat_df], axis=1)

    # Target
    if 'Delivery_Time' in df.columns:
        df['Delivery_Time'] = pd.to_numeric(df['Delivery_Time'], errors='coerce')
        df = df.dropna(subset=['Delivery_Time'])

    # Final features
    feature_cols = [c for c in ['Agent_Age','Agent_Rating','Distance_km','Pickup_Lag_min','Hour','DayOfWeek'] if c in df.columns]
    feature_cols += [c + '_enc' for c in cat_cols]

    return df[feature_cols + ['Delivery_Time']]

def main(input_csv, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    processed = preprocess(input_csv)
    out_path = os.path.join(out_dir, 'processed_amazon_delivery.csv')
    processed.to_csv(out_path, index=False)
    print('✅ Saved processed data to', out_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', required=True)
    parser.add_argument('--out_dir', default='data')
    args = parser.parse_args()
    main(args.input_csv, args.out_dir)
