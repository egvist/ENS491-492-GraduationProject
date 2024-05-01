from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import math


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
    branch_name = get_branch_name(branch_no)  # You need to implement this function
    # Preprocess data
    weekly_data = branch_data.resample('W-Mon').size()
    train_size = int(len(weekly_data) * 0.8)
    train_data, test_data = weekly_data.iloc[:train_size+1], weekly_data.iloc[train_size:]

    # Train ETS model
    ets_model = ExponentialSmoothing(train_data, seasonal_periods=12, trend='add', seasonal='add').fit()

    # Forecast using ETS model
    forecast_steps = len(test_data) + 5  
    forecast_ets = ets_model.forecast(steps=forecast_steps)
    first = forecast_ets[1:2].tolist()

    return {"branch_name": branch_name, "forecast": first}

@app.get('/forecast_total')
def forecast_total():
    if df.empty:
        raise HTTPException(status_code=400, detail="Please upload an Excel file first.")
    # Preprocess data
    weekly_data = df.resample('W-Mon').size()
    train_size = int(len(weekly_data) * 0.8)
    train_data, test_data = weekly_data.iloc[:train_size+1], weekly_data.iloc[train_size:]

    # Train ETS model
    ets_model = ExponentialSmoothing(train_data, seasonal_periods=12, trend='add', seasonal='add').fit()

    # Forecast using ETS model
    forecast_steps = len(test_data) + 5  
    forecast_ets = ets_model.forecast(steps=forecast_steps)
    first = forecast_ets[1:2].tolist()

    return {"forecast": first}





@app.get('/forecast_employee/{branch_no}')
def forecast_employee(branch_no: int):
    if df.empty:
        raise HTTPException(status_code=400, detail="Please upload an Excel file first.")

    branch_data = df[df['Servis Noktası'] == branch_no].copy()
    branch_name = get_branch_name(branch_no)
    # Preprocess data
    weekly_data = branch_data.resample('W-Mon').size()
    train_size = int(len(weekly_data) * 0.8)
    train_data, test_data = weekly_data.iloc[:train_size+1], weekly_data.iloc[train_size:]

    # Train ETS model
    ets_model = ExponentialSmoothing(train_data, seasonal_periods=12, trend='add', seasonal='add').fit()

    # Forecast using ETS model
    forecast_steps = len(test_data) + 5  
    forecast_ets = ets_model.forecast(steps=forecast_steps)
    
    # Calculate average forecasted cases for the next week
    forecasted_total_cases_next_week = forecast_ets[1:2]
    forecasted_total_cases_next_week = forecasted_total_cases_next_week.reset_index(drop=True)
    forecasted_total_cases_next_week.index = forecasted_total_cases_next_week.index.astype('int64')

    category_proportions = df['Model Kategorisi'].value_counts(normalize=True)
    category_proportions = category_proportions.reset_index(drop=True)


    # Now the indices should match, and we can perform the multiplication
    estimated_cases_by_category = forecasted_total_cases_next_week[0] * category_proportions

    # Create DataFrame for estimated distribution
    estimated_distribution_df = estimated_cases_by_category.reset_index()
    estimated_distribution_df.columns = ['Model Kategorisi', 'Gelecek Hafta Tahmini Arıza Sayısı']
    estimated_distribution_df['Model Kategorisi'] = pd.to_numeric(estimated_distribution_df['Model Kategorisi'], errors='coerce')
    estimated_distribution_df.sort_values(by='Model Kategorisi', ascending=True, inplace=True)
    estimated_distribution_df.reset_index(drop=True, inplace=True)

    # Define minutes per case for each category
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
    
    # Calculate estimated minutes for each category
    estimated_distribution_df['Gelecek Hafta Tahmini Dakikalar'] = estimated_distribution_df.apply(
        lambda row: row['Gelecek Hafta Tahmini Arıza Sayısı'] * minutes_per_case.get(row['Model Kategorisi'], 0), axis=1)
    # Calculate total estimated minutes
    total_estimated_minutes = estimated_distribution_df['Gelecek Hafta Tahmini Dakikalar'].sum()

    # Calculate number of employees
    calculated_number_of_employees = total_estimated_minutes / 2880  # Assuming 2880 minutes per week per employee
    calculated_number_of_employees = math.ceil(calculated_number_of_employees)  # Round up to nearest integer

    return {"branch_name": branch_name, "estimated_number_of_employees": calculated_number_of_employees}




@app.get('/forecast_total_employee')
def forecast_total_employee():
    if df.empty:
        raise HTTPException(status_code=400, detail="Please upload an Excel file first.")

    # Preprocess data
    weekly_data = df.resample('W-Mon').size()
    train_size = int(len(weekly_data) * 0.8)
    train_data, test_data = weekly_data.iloc[:train_size+1], weekly_data.iloc[train_size:]

    # Train ETS model
    ets_model = ExponentialSmoothing(train_data, seasonal_periods=12, trend='add', seasonal='add').fit()

    # Forecast using ETS model
    forecast_steps = len(test_data) + 5  
    forecast_ets = ets_model.forecast(steps=forecast_steps)
    
    # Calculate average forecasted cases for the next week
    forecasted_total_cases_next_week = forecast_ets[1:2]
    forecasted_total_cases_next_week = forecasted_total_cases_next_week.reset_index(drop=True)
    forecasted_total_cases_next_week.index = forecasted_total_cases_next_week.index.astype('int64')

    category_proportions = df['Model Kategorisi'].value_counts(normalize=True)
    category_proportions = category_proportions.reset_index(drop=True)


    # Now the indices should match, and we can perform the multiplication
    estimated_cases_by_category = forecasted_total_cases_next_week[0] * category_proportions

    # Create DataFrame for estimated distribution
    estimated_distribution_df = estimated_cases_by_category.reset_index()
    estimated_distribution_df.columns = ['Model Kategorisi', 'Gelecek Hafta Tahmini Arıza Sayısı']
    estimated_distribution_df['Model Kategorisi'] = pd.to_numeric(estimated_distribution_df['Model Kategorisi'], errors='coerce')
    estimated_distribution_df.sort_values(by='Model Kategorisi', ascending=True, inplace=True)
    estimated_distribution_df.reset_index(drop=True, inplace=True)

    # Define minutes per case for each category
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
    
    # Calculate estimated minutes for each category
    estimated_distribution_df['Gelecek Hafta Tahmini Dakikalar'] = estimated_distribution_df.apply(
        lambda row: row['Gelecek Hafta Tahmini Arıza Sayısı'] * minutes_per_case.get(row['Model Kategorisi'], 0), axis=1)
    
    # Calculate total estimated minutes
    total_estimated_minutes = estimated_distribution_df['Gelecek Hafta Tahmini Dakikalar'].sum()

    # Calculate number of employees
    calculated_number_of_employees = total_estimated_minutes / 2880  # Assuming 2880 minutes per week per employee

    calculated_number_of_employees = math.ceil(calculated_number_of_employees)  # Round up to nearest integer



    return {"estimated_number_of_employees": calculated_number_of_employees}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
