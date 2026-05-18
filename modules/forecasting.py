import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from pyspark.sql import functions as F

def run_arima(df, output_dir):
    from statsmodels.tsa.arima.model import ARIMA
    from sklearn.metrics import mean_squared_error

    os.makedirs(output_dir, exist_ok=True)

    daily_sales = df.groupBy('order_dow').agg(
        F.countDistinct('order_id').alias('num_orders'),
        F.count('product_id').alias('total_items')
    ).orderBy('order_dow').toPandas()

    n_weeks = 52
    np.random.seed(42)
    ts_data = []
    base_date = pd.Timestamp('2023-01-02')
    for w in range(n_weeks):
        for _, row in daily_sales.iterrows():
            date = base_date + pd.Timedelta(weeks=w, days=int(row['order_dow']))
            noise = np.random.normal(0, row['total_items'] * 0.05)
            ts_data.append({'date': date, 'sales': max(0, row['total_items'] + noise)})

    ts_df = pd.DataFrame(ts_data).sort_values('date').reset_index(drop=True)
    ts_df.set_index('date', inplace=True)
    weekly = ts_df.resample('W').sum()
    print(f'Weekly time series: {len(weekly)} points')

    train_size = int(len(weekly) * 0.8)
    train_ts = weekly[:train_size]
    test_ts = weekly[train_size:]

    model = ARIMA(train_ts['sales'], order=(2, 1, 2))
    fit = model.fit()
    forecast = fit.forecast(steps=len(test_ts))
    rmse = np.sqrt(mean_squared_error(test_ts['sales'], forecast))
    print(f'ARIMA RMSE: {rmse:.4f}')

    plt.figure(figsize=(14, 5))
    plt.plot(weekly.index, weekly['sales'], 'b-')
    plt.title('Weekly Sales')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'timeseries_weekly.png'), dpi=100)
    plt.close()

    plt.figure(figsize=(14, 5))
    plt.plot(train_ts.index, train_ts['sales'], 'b-', label='Train')
    plt.plot(test_ts.index, test_ts['sales'], 'g-', label='Actual')
    plt.plot(test_ts.index, forecast, 'r--', label=f'ARIMA (RMSE={rmse:.2f})')
    plt.title('ARIMA Forecasting')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'timeseries_arima.png'), dpi=100)
    plt.close()

    return rmse
