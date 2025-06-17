import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error
from crewai.tools import tool
import matplotlib.pyplot as plt
import os
from io import StringIO

@tool("Sales Forecasting and Insights Extraction Tool")
def forecast_and_analyze_sales(cleaned_data_path: str) -> dict:
    """
    Takes the file path of cleaned store-level sales data, trains a 
    Prophet model, creates a 52-week forecast, and correctly evaluates 
    performance on a 52-week holdout set using actual regressor values.
    """
    try:
        data = pd.read_csv(cleaned_data_path)
    except FileNotFoundError:
        return f"Error: The data file was not found at path: {cleaned_data_path}"

    data['ds'] = pd.to_datetime(data['ds'])
    data['y'] = pd.to_numeric(data['y'], errors='coerce')
    data.dropna(subset=['y'], inplace=True)
    
    regressors = ['Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
    
    # --- 1. Evaluation using a 52-week holdout set ---
    split_date = data['ds'].max() - pd.DateOffset(weeks=52)
    train_df = data[data['ds'] <= split_date]
    test_df = data[data['ds'] > split_date]

    # This model is trained only on the training data
    model_for_evaluation = Prophet()
    for reg in regressors:
        model_for_evaluation.add_regressor(reg)
    model_for_evaluation.fit(train_df)
    

    forecast_eval = model_for_evaluation.predict(test_df)

    # Calculate MAPE on the evaluation forecast
    y_true = test_df['y']
    y_pred = forecast_eval['yhat']
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100

    # --- 2. Final Model Training on ALL Data ---

    final_model = Prophet()
    for reg in regressors:
        final_model.add_regressor(reg)
    final_model.fit(data)

    # --- 3. 52-Week Future Forecasting ---
    future_final = final_model.make_future_dataframe(periods=52, freq='W')
    
    # For the true future, we use the last known values as a reasonable proxy
    last_known_regressors = data.iloc[-1][regressors]
    for col in regressors:
        future_final[col] = last_known_regressors[col]
    
    final_forecast = final_model.predict(future_final)

    # --- 4. Prepare Final Results ---
    regressor_names = list(final_model.extra_regressors.keys())
    regressor_coefficients = {name: np.mean(final_model.params['beta'], axis=0)[i] for i, name in enumerate(regressor_names)}
    coefs_markdown = "\n".join([f"- **{name}:** `{value:.2f}`" for name, value in regressor_coefficients.items()])

    forecasted_period = final_forecast.iloc[-52:]
    forecast_total = int(forecasted_period['yhat'].sum())
    avg_weekly = int(forecasted_period['yhat'].mean())
    
    os.makedirs('outputs', exist_ok=True)
    fig_forecast = final_model.plot(final_forecast)
    forecast_plot_path = f'outputs/store_{data["Store"].iloc[0]}_forecast_plot.png'
    fig_forecast.savefig(forecast_plot_path)
    plt.close(fig_forecast)

    results = {
        "mape": round(mape, 2),
        "forecast_total": forecast_total,
        "avg_weekly": avg_weekly,
        "regressor_coefficients": coefs_markdown,
        "forecast_plot_path": forecast_plot_path
    }
    return results