from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import math
from pmdarima import auto_arima
from tensorflow.keras.layers import GRU

app = FastAPI()


class FileUploadResponse(BaseModel):
    filename: str

# Load and preprocess data
#default_df = pd.read_excel("Sabancı Servis Verisi.xlsx")
#df = default_df.copy() 
#df['Date'] = pd.to_datetime(df['Kaydı Üzerine Alma Tarihi'])  # Convert to datetime
#df.set_index('Date', inplace=True)  # Set 'Date' column as the index
branches = [11, 12, 13, 14, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 46]

branch_names = {
    11: "İstanbul Destek Kadıköy",
    12: "İstanbul KVK Pendik",
    13:"Ankara Arım",
    14:"Ankara Simge",
    16:"Antalya Yıldırım",
    17:"İstanbul MTA Bakırköy",
    18:"İstanbul Destek GOP",
    19:"İstanbul KVK Şişli",
    20:"Ankara Destek",
    22:"İzmir Destek",
    23:"İstanbul Başarı Avcılar",
    24:"Diyarbakır Hizmet",
    25:"Bursa Ebru",
    26:"İstanbul Danıştek Ümraniye",
    27:"Adana Destek",
    28:"Samsun Bayrak",
    29:"Ankara Başarı",
    30:"Başarı Adana",
    31:"İzmir Başarı",
    46:"Sancaktepe Destek"

}

minutes_per_case = {
    0: 81,
    1: 71,
    2: 47,
    3: 42,
    4: 55,
    5: 39,
    6: 56,
    7: 56,
    8: 56,
    9:56
}

def get_branch_name(branch_no):
    # Check if the branch number exists in the dictionary
    if branch_no in branch_names:
        return branch_names[branch_no]
    else:
        return "Bilinmeyen Şube"  # Default name if branch number not found
    
# LSTM Model
def create_dataset(data, time_steps=1):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps), 0])
        y.append(data[i + time_steps, 0])
    return np.array(X), np.array(y)

scaler = MinMaxScaler(feature_range=(0, 1))

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    if file.filename.endswith(".xlsx"):
        contents = await file.read()  # Read the contents of the uploaded file
        global df
        df = pd.read_excel(contents)  # Load the uploaded Excel file into a DataFrame
        df['Date'] = pd.to_datetime(df['Kaydı Üzerine Alma Tarihi'])  # Convert to datetime
        df.set_index('Date', inplace=True)  # Set 'Date' column as the index
        return {"filename": file.filename}
    else:
        raise HTTPException(status_code=400, detail="Only Excel files (.xlsx) are allowed.")



# API endpoint for forecasting
@app.get('/forecast/{branch_no}')
def forecast(branch_no: int):
    if df.empty:
        raise HTTPException(status_code=400, detail="Please upload an Excel file first.")

    branch_data = df[df['Servis Noktası'] == branch_no].copy()
    branch_name = get_branch_name(branch_no)
    weekly_data = branch_data.resample('W-Mon').size()
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(weekly_data.values.reshape(-1, 1))
    time_steps = 24
    X, y = create_dataset(scaled_data, time_steps)
    test_size = int(len(X) * 0.2)
    X_train, X_test, y_train, y_test = X[:-test_size], X[-test_size:], y[:-test_size], y[-test_size:]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))


    #ETS
    train_size = int(len(weekly_data) * 0.8)
    train_data, test_data = weekly_data.iloc[:train_size+1], weekly_data.iloc[train_size:]
    ets_model = ExponentialSmoothing(train_data, seasonal_periods=12, trend='add', seasonal='add').fit()
    forecast_steps = len(test_data) + 5  
    forecast_ets = ets_model.forecast(steps=forecast_steps)

    #ARIMA
    arima_model = auto_arima(train_data, start_p=1, start_q=1,
                                max_p=5, max_q=5, d=1,
                                seasonal=False, trace=True,
                                error_action='ignore',
                                suppress_warnings=True,
                                stepwise=True)
    arima_model.fit(train_data)
    forecast_arima = arima_model.predict(n_periods=len(test_data) + 5)
    arima_rmse = math.sqrt(mean_squared_error(test_data, forecast_arima[:len(test_data)]))

    #GRU
    gru_model = Sequential([
        GRU(50, return_sequences=True, input_shape=(time_steps, 1)),
        Dropout(0.2),
        GRU(50),
        Dropout(0.2),
        Dense(1)
    ])
    gru_model.compile(optimizer='adam', loss='mean_squared_error')
    gru_model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
    gru_test_predictions = gru_model.predict(X_test)
    gru_test_predictions = scaler.inverse_transform(gru_test_predictions)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    input_seq = scaled_data[-time_steps:].reshape(1, time_steps, 1)
    forecasted_values = []
    for _ in range(15):
        pred = gru_model.predict(input_seq)
        forecasted_values.append(pred[0, 0])
        input_seq = np.append(input_seq[:, 1:, :], pred.reshape(1, 1, 1), axis=1)
    gru_forecasted_values = scaler.inverse_transform(np.array(forecasted_values).reshape(-1, 1))
    gru_rmse = math.sqrt(mean_squared_error(y_test_inv, gru_test_predictions))

    #LSTM
    lstm_model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(time_steps, 1)),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    lstm_model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
    test_predictions = lstm_model.predict(X_test)
    test_predictions = scaler.inverse_transform(test_predictions)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    input_seq = scaled_data[-time_steps:].reshape(1, time_steps, 1)
    forecasted_values = []
    for _ in range(15):
        pred = lstm_model.predict(input_seq)
        forecasted_values.append(pred[0, 0])
        input_seq = np.append(input_seq[:, 1:, :], pred.reshape(1, 1, 1), axis=1)
    forecasted_values = scaler.inverse_transform(np.array(forecasted_values).reshape(-1, 1))
 
    ets_rmse = math.sqrt(mean_squared_error(test_data, forecast_ets[:len(test_data)]))
    lstm_rmse = math.sqrt(mean_squared_error(y_test_inv, test_predictions))

    selected_model = "ETS"
    forecasted_values = forecast_ets[-6:]  # Use ETS forecast by default
    if arima_rmse < ets_rmse and arima_rmse < lstm_rmse and arima_rmse < gru_rmse:
        selected_model = "ARIMA"
        forecasted_values = forecast_arima[-6:]  # Use ARIMA forecast if its RMSE is the lowest
    elif lstm_rmse < ets_rmse and lstm_rmse < gru_rmse:
        selected_model = "LSTM"
        forecasted_values = forecasted_values[:5]  # Use LSTM forecast if its RMSE is the lowest
    elif gru_rmse < ets_rmse and gru_rmse < lstm_rmse:
        selected_model = "GRU"
        forecasted_values = gru_forecasted_values[:5]  # Use GRU forecast if its RMSE is the lowest

    # Calculate average forecasted cases for the next week
    forecasted_total_cases_next_week = forecasted_values[:1].tolist()

    return {"branch_name": branch_name, "selected_model": selected_model,"ets_rmse": ets_rmse, "lstm_rmse": lstm_rmse, "gru_rmse": gru_rmse, "arima_rmse": arima_rmse, "forecast": forecasted_total_cases_next_week}

@app.get('/forecast_total')
def forecast_total():
    if df.empty:
        raise HTTPException(status_code=400, detail="Please upload an Excel file first.")
    # Preprocess data
    weekly_data = df.resample('W-Mon').size()
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(weekly_data.values.reshape(-1, 1))
    time_steps = 24
    X, y = create_dataset(scaled_data, time_steps)
    test_size = int(len(X) * 0.2)
    X_train, X_test, y_train, y_test = X[:-test_size], X[-test_size:], y[:-test_size], y[-test_size:]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))


    train_size = int(len(weekly_data) * 0.8)
    train_data, test_data = weekly_data.iloc[:train_size+1], weekly_data.iloc[train_size:]
    ets_model = ExponentialSmoothing(train_data, seasonal_periods=12, trend='add', seasonal='add').fit()
    forecast_steps = len(test_data) + 5  
    forecast_ets = ets_model.forecast(steps=forecast_steps)

    #ARIMA
    arima_model = auto_arima(train_data, start_p=1, start_q=1,
                                max_p=5, max_q=5, d=1,
                                seasonal=False, trace=True,
                                error_action='ignore',
                                suppress_warnings=True,
                                stepwise=True)
    arima_model.fit(train_data)
    forecast_arima = arima_model.predict(n_periods=len(test_data) + 5)
    arima_rmse = math.sqrt(mean_squared_error(test_data, forecast_arima[:len(test_data)]))
    #GRU
    gru_model = Sequential([
        GRU(50, return_sequences=True, input_shape=(time_steps, 1)),
        Dropout(0.2),
        GRU(50),
        Dropout(0.2),
        Dense(1)
    ])
    gru_model.compile(optimizer='adam', loss='mean_squared_error')
    gru_model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
    gru_test_predictions = gru_model.predict(X_test)
    gru_test_predictions = scaler.inverse_transform(gru_test_predictions)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    input_seq = scaled_data[-time_steps:].reshape(1, time_steps, 1)
    forecasted_values = []
    for _ in range(15):
        pred = gru_model.predict(input_seq)
        forecasted_values.append(pred[0, 0])
        input_seq = np.append(input_seq[:, 1:, :], pred.reshape(1, 1, 1), axis=1)
    gru_forecasted_values = scaler.inverse_transform(np.array(forecasted_values).reshape(-1, 1))
    gru_rmse = math.sqrt(mean_squared_error(y_test_inv, gru_test_predictions))

    
    #LSTM
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(weekly_data.values.reshape(-1, 1))
    time_steps = 24
    X, y = create_dataset(scaled_data, time_steps)
    test_size = int(len(X) * 0.2)
    X_train, X_test, y_train, y_test = X[:-test_size], X[-test_size:], y[:-test_size], y[-test_size:]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    lstm_model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(time_steps, 1)),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    lstm_model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
    test_predictions = lstm_model.predict(X_test)
    test_predictions = scaler.inverse_transform(test_predictions)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    input_seq = scaled_data[-time_steps:].reshape(1, time_steps, 1)
    forecasted_values = []
    for _ in range(15):
        pred = lstm_model.predict(input_seq)
        forecasted_values.append(pred[0, 0])
        input_seq = np.append(input_seq[:, 1:, :], pred.reshape(1, 1, 1), axis=1)
    forecasted_values = scaler.inverse_transform(np.array(forecasted_values).reshape(-1, 1))
 
    ets_rmse = math.sqrt(mean_squared_error(test_data, forecast_ets[:len(test_data)]))
    lstm_rmse = math.sqrt(mean_squared_error(y_test_inv, test_predictions))

    selected_model = "ETS"
    forecasted_values = forecast_ets[-6:]  # Use ETS forecast by default
    if arima_rmse < ets_rmse and arima_rmse < lstm_rmse and arima_rmse < gru_rmse:
        selected_model = "ARIMA"
        forecasted_values = forecast_arima[-6:]  # Use ARIMA forecast if its RMSE is the lowest
    elif lstm_rmse < ets_rmse and lstm_rmse < gru_rmse:
        selected_model = "LSTM"
        forecasted_values = forecasted_values[:5]  # Use LSTM forecast if its RMSE is the lowest
    elif gru_rmse < ets_rmse and gru_rmse < lstm_rmse:
        selected_model = "GRU"
        forecasted_values = gru_forecasted_values[:5]  # Use GRU forecast if its RMSE is the lowest

    # Calculate average forecasted cases for the next week
    forecasted_total_cases_next_week = forecasted_values[:1].tolist()

    return {"selected_model": selected_model,"ets_rmse": ets_rmse, "lstm_rmse": lstm_rmse, "gru_rmse": gru_rmse, "arima_rmse": arima_rmse, "forecast": forecasted_total_cases_next_week}






@app.get('/forecast_employee/{branch_no}')
def forecast_employee(branch_no: int):
    if df.empty:
        raise HTTPException(status_code=400, detail="Please upload an Excel file first.")

    
    branch_data = df[df['Servis Noktası'] == branch_no].copy()
    branch_name = get_branch_name(branch_no)
    weekly_data = branch_data.resample('W-Mon').size()
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(weekly_data.values.reshape(-1, 1))
    time_steps = 24
    X, y = create_dataset(scaled_data, time_steps)
    test_size = int(len(X) * 0.2)
    X_train, X_test, y_train, y_test = X[:-test_size], X[-test_size:], y[:-test_size], y[-test_size:]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    #ETS
    train_size = int(len(weekly_data) * 0.8)
    train_data, test_data = weekly_data.iloc[:train_size+1], weekly_data.iloc[train_size:]
    ets_model = ExponentialSmoothing(train_data, seasonal_periods=12, trend='add', seasonal='add').fit()
    forecast_steps = len(test_data) + 5  
    forecast_ets = ets_model.forecast(steps=forecast_steps)

    #ARIMA
    arima_model = auto_arima(train_data, start_p=1, start_q=1,
                                max_p=5, max_q=5, d=1,
                                seasonal=False, trace=True,
                                error_action='ignore',
                                suppress_warnings=True,
                                stepwise=True)
    arima_model.fit(train_data)
    forecast_arima = arima_model.predict(n_periods=len(test_data) + 5)
    arima_rmse = math.sqrt(mean_squared_error(test_data, forecast_arima[:len(test_data)]))
    #GRU
    gru_model = Sequential([
        GRU(50, return_sequences=True, input_shape=(time_steps, 1)),
        Dropout(0.2),
        GRU(50),
        Dropout(0.2),
        Dense(1)
    ])
    gru_model.compile(optimizer='adam', loss='mean_squared_error')
    gru_model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
    gru_test_predictions = gru_model.predict(X_test)
    gru_test_predictions = scaler.inverse_transform(gru_test_predictions)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    input_seq = scaled_data[-time_steps:].reshape(1, time_steps, 1)
    forecasted_values = []
    for _ in range(15):
        pred = gru_model.predict(input_seq)
        forecasted_values.append(pred[0, 0])
        input_seq = np.append(input_seq[:, 1:, :], pred.reshape(1, 1, 1), axis=1)
    gru_forecasted_values = scaler.inverse_transform(np.array(forecasted_values).reshape(-1, 1))
    gru_rmse = math.sqrt(mean_squared_error(y_test_inv, gru_test_predictions))

    #LSTM
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(weekly_data.values.reshape(-1, 1))
    time_steps = 24
    X, y = create_dataset(scaled_data, time_steps)
    test_size = int(len(X) * 0.2)
    X_train, X_test, y_train, y_test = X[:-test_size], X[-test_size:], y[:-test_size], y[-test_size:]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    lstm_model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(time_steps, 1)),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    lstm_model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
    test_predictions = lstm_model.predict(X_test)
    test_predictions = scaler.inverse_transform(test_predictions)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    input_seq = scaled_data[-time_steps:].reshape(1, time_steps, 1)
    forecasted_values = []
    for _ in range(15):
        pred = lstm_model.predict(input_seq)
        forecasted_values.append(pred[0, 0])
        input_seq = np.append(input_seq[:, 1:, :], pred.reshape(1, 1, 1), axis=1)
    forecasted_values = scaler.inverse_transform(np.array(forecasted_values).reshape(-1, 1))
 
    ets_rmse = math.sqrt(mean_squared_error(test_data, forecast_ets[:len(test_data)]))
    lstm_rmse = math.sqrt(mean_squared_error(y_test_inv, test_predictions))

    selected_model = "ETS"
    forecasted_values = forecast_ets[-6:]  # Use ETS forecast by default
    if arima_rmse < ets_rmse and arima_rmse < lstm_rmse and arima_rmse < gru_rmse:
        selected_model = "ARIMA"
        forecasted_values = forecast_arima[-6:]  # Use ARIMA forecast if its RMSE is the lowest
    elif lstm_rmse < ets_rmse and lstm_rmse < gru_rmse:
        selected_model = "LSTM"
        forecasted_values = forecasted_values[:5]  # Use LSTM forecast if its RMSE is the lowest
    elif gru_rmse < ets_rmse and gru_rmse < lstm_rmse:
        selected_model = "GRU"
        forecasted_values = gru_forecasted_values[:5]  # Use GRU forecast if its RMSE is the lowest

    # Calculate average forecasted cases for the next week
    forecasted_total_cases_next_week = forecasted_values[:1]
    forecasted_total_cases_next_week_df = pd.DataFrame(forecasted_total_cases_next_week)
    forecasted_total_cases_next_week_df.reset_index(drop=True, inplace=True)
    forecasted_total_cases_next_week_df.index = forecasted_total_cases_next_week_df.index.astype('int64')


    category_counts = df['Model Kategorisi'].value_counts()
    category_proportions = category_counts / category_counts.sum()
    category_proportions = category_proportions.reset_index(drop=True)


    # Ensure the indexes of forecasted_total_cases_next_week_df[0] and category_proportions align
    # Ensure the indexes of forecasted_total_cases_next_week_df[0] and category_proportions align
    forecasted_total_cases_next_week_df[0] = forecasted_total_cases_next_week_df[0].reindex(category_proportions.index, fill_value=0)

    # Multiply forecasted cases with category proportions
    estimated_cases_by_category = []
    for index, proportion in category_proportions.items():
        estimated_cases = forecasted_total_cases_next_week_df[0] * proportion
        # Append a DataFrame row to the list
        estimated_cases_by_category.append([index, estimated_cases[0]])


    # Create a DataFrame from the list
    estimated_cases_by_category_df = pd.DataFrame(estimated_cases_by_category)

    # Set column names
    estimated_cases_by_category_df.columns = ['Index', 'Estimated_Cases']


    # Reset the index and rename the index column
    estimated_cases_by_category_df.reset_index(drop=True, inplace=True)
    estimated_cases_by_category_df.rename(columns={'index': 'Model_Kategorisi'}, inplace=True)

    estimated_distribution_df = estimated_cases_by_category_df
    estimated_distribution_df.columns = ['Model Kategorisi', 'Gelecek Hafta Tahmini Arıza Sayısı']
    estimated_distribution_df['Model Kategorisi'] = pd.to_numeric(estimated_distribution_df['Model Kategorisi'], errors='coerce')
    estimated_distribution_df.sort_values(by='Model Kategorisi', ascending=True, inplace=True)
    estimated_distribution_df.reset_index(drop=True, inplace=True)
    estimated_distribution_df['Gelecek Hafta Tahmini Dakikalar'] = estimated_distribution_df.apply(
        lambda row: row['Gelecek Hafta Tahmini Arıza Sayısı'] * minutes_per_case.get(row['Model Kategorisi'], 0), axis=1)
    total_estimated_minutes = estimated_distribution_df['Gelecek Hafta Tahmini Dakikalar'].sum()
    calculated_number_of_employees = total_estimated_minutes / 2880  # Assuming 2880 minutes per week per employee
    calculated_number_of_employees = math.ceil(calculated_number_of_employees)  # Round up to nearest integer

    return {"branch_name": branch_name, "selected_model": selected_model, "ets_rmse": ets_rmse, "lstm_rmse": lstm_rmse, "gru_rmse": gru_rmse, "arima_rmse": arima_rmse,"estimated_number_of_employees": calculated_number_of_employees}


@app.get('/forecast_total_employee')
def forecast_total_employee():
    if df.empty:
        raise HTTPException(status_code=400, detail="Please upload an Excel file first.")

    
    weekly_data = df.resample('W-Mon').size()
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(weekly_data.values.reshape(-1, 1))
    time_steps = 24
    X, y = create_dataset(scaled_data, time_steps)
    test_size = int(len(X) * 0.2)
    X_train, X_test, y_train, y_test = X[:-test_size], X[-test_size:], y[:-test_size], y[-test_size:]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    #ETS
    train_size = int(len(weekly_data) * 0.8)
    train_data, test_data = weekly_data.iloc[:train_size+1], weekly_data.iloc[train_size:]
    ets_model = ExponentialSmoothing(train_data, seasonal_periods=12, trend='add', seasonal='add').fit()
    forecast_steps = len(test_data) + 5  
    forecast_ets = ets_model.forecast(steps=forecast_steps)
    
    #ARIMA
    arima_model = auto_arima(train_data, start_p=1, start_q=1,
                                max_p=5, max_q=5, d=1,
                                seasonal=False, trace=True,
                                error_action='ignore',
                                suppress_warnings=True,
                                stepwise=True)
    arima_model.fit(train_data)
    forecast_arima = arima_model.predict(n_periods=len(test_data) + 5)
    arima_rmse = math.sqrt(mean_squared_error(test_data, forecast_arima[:len(test_data)]))

    #GRU
    gru_model = Sequential([
        GRU(50, return_sequences=True, input_shape=(time_steps, 1)),
        Dropout(0.2),
        GRU(50),
        Dropout(0.2),
        Dense(1)
    ])
    gru_model.compile(optimizer='adam', loss='mean_squared_error')
    gru_model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
    gru_test_predictions = gru_model.predict(X_test)
    gru_test_predictions = scaler.inverse_transform(gru_test_predictions)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    input_seq = scaled_data[-time_steps:].reshape(1, time_steps, 1)
    forecasted_values = []
    for _ in range(15):
        pred = gru_model.predict(input_seq)
        forecasted_values.append(pred[0, 0])
        input_seq = np.append(input_seq[:, 1:, :], pred.reshape(1, 1, 1), axis=1)
    gru_forecasted_values = scaler.inverse_transform(np.array(forecasted_values).reshape(-1, 1))
    gru_rmse = math.sqrt(mean_squared_error(y_test_inv, gru_test_predictions))

    #LSTM
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(weekly_data.values.reshape(-1, 1))
    time_steps = 24
    X, y = create_dataset(scaled_data, time_steps)
    test_size = int(len(X) * 0.2)
    X_train, X_test, y_train, y_test = X[:-test_size], X[-test_size:], y[:-test_size], y[-test_size:]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    lstm_model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(time_steps, 1)),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    lstm_model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
    test_predictions = lstm_model.predict(X_test)
    test_predictions = scaler.inverse_transform(test_predictions)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    input_seq = scaled_data[-time_steps:].reshape(1, time_steps, 1)
    forecasted_values = []
    for _ in range(15):
        pred = lstm_model.predict(input_seq)
        forecasted_values.append(pred[0, 0])
        input_seq = np.append(input_seq[:, 1:, :], pred.reshape(1, 1, 1), axis=1)
    forecasted_values = scaler.inverse_transform(np.array(forecasted_values).reshape(-1, 1))
 

    ets_rmse = math.sqrt(mean_squared_error(test_data, forecast_ets[:len(test_data)]))
    lstm_rmse = math.sqrt(mean_squared_error(y_test_inv, test_predictions))

    selected_model = "ETS"
    forecasted_values = forecast_ets[-6:]  # Use ETS forecast by default
    if arima_rmse < ets_rmse and arima_rmse < lstm_rmse and arima_rmse < gru_rmse:
        selected_model = "ARIMA"
        forecasted_values = forecast_arima[-6:]  # Use ARIMA forecast if its RMSE is the lowest
    elif lstm_rmse < ets_rmse and lstm_rmse < gru_rmse:
        selected_model = "LSTM"
        forecasted_values = forecasted_values[:5]  # Use LSTM forecast if its RMSE is the lowest
    elif gru_rmse < ets_rmse and gru_rmse < lstm_rmse:
        selected_model = "GRU"
        forecasted_values = gru_forecasted_values[:5]  # Use GRU forecast if its RMSE is the lowest


    # Calculate average forecasted cases for the next week
    forecasted_total_cases_next_week = forecasted_values[:1]
    forecasted_total_cases_next_week = forecasted_total_cases_next_week.reset_index(drop=True)
    forecasted_total_cases_next_week.index = forecasted_total_cases_next_week.index.astype('int64')

    category_proportions = df['Model Kategorisi'].value_counts(normalize=True)
    category_proportions = category_proportions.reset_index(drop=True)

    estimated_cases_by_category = forecasted_total_cases_next_week[0] * category_proportions
    estimated_distribution_df = estimated_cases_by_category.reset_index()
    estimated_distribution_df.columns = ['Model Kategorisi', 'Gelecek Hafta Tahmini Arıza Sayısı']
    estimated_distribution_df['Model Kategorisi'] = pd.to_numeric(estimated_distribution_df['Model Kategorisi'], errors='coerce')
    estimated_distribution_df.sort_values(by='Model Kategorisi', ascending=True, inplace=True)
    estimated_distribution_df.reset_index(drop=True, inplace=True)
    estimated_distribution_df['Gelecek Hafta Tahmini Dakikalar'] = estimated_distribution_df.apply(
        lambda row: row['Gelecek Hafta Tahmini Arıza Sayısı'] * minutes_per_case.get(row['Model Kategorisi'], 0), axis=1)
    total_estimated_minutes = estimated_distribution_df['Gelecek Hafta Tahmini Dakikalar'].sum()
    calculated_number_of_employees = total_estimated_minutes / 2880  # Assuming 2880 minutes per week per employee
    calculated_number_of_employees = math.ceil(calculated_number_of_employees)  # Round up to nearest integer


    return {"estimated_number_of_employees": calculated_number_of_employees}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
