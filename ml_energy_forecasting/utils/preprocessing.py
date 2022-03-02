import pandas as pd

def load_data(demand_path, temperature_path, holiday_path):
  out = {
    'demand_df': pd.read_csv(demand_path),
    'temperature_df': pd.read_csv(temperature_path),
    'holiday_df': pd.read_csv(holiday_path, parse_dates=[0], header=None, names=['Date'])
  }
  
  return out

def preprocess_and_merge(demand_df, temperature_df, holiday_df, normalise_to_today=False):
  # Convert the date index to an actual date
  demand_df['Date'] = pd.to_datetime('1899-12-30') + pd.Timedelta('1 day') * demand_df['Date']
  demand_df.drop(['Period','Industrial'], axis=1, inplace=True)

  # Now filter for the 2002-2014 data and aggregate by day
  demand_df = demand_df.loc[demand_df.Date < pd.to_datetime('2015-01-01'), :]

  # Convert the date index to an actual date
  temperature_df['Date'] = pd.to_datetime('1899-12-30') + pd.Timedelta('1 day') * temperature_df['Date']
  temperature_df = temperature_df.loc[temperature_df.Period==30, :]
  temperature_df.drop(['Period'], axis=1, inplace=True)

  # Create dummy column
  holiday_df['holiday'] = 1

  # Aggegate to daily data
  df = demand_df.groupby('Date').sum().reset_index()  
  
  # Finally, merge
  df = df \
    .merge(temperature_df, on='Date', how='left') \
    .merge(holiday_df, on='Date', how='left')
  
  df.loc[pd.isnull(df['holiday']), 'holiday'] = 0   # not a holiday
  df.columns = ['ds', 'y', 'temp', 'holiday']
  
  # Temperate squared, a feature
  df['temp_2'] = df['temp']**2
  
  if normalise_to_today:
    df['ds'] = df['ds'] + (datetime.datetime.now() - df['ds'].max())
    
  return df

def preprocess_demand_df(df):
  df['Date'] = pd.to_datetime('1899-12-30') + pd.Timedelta('1 day') * df['Date']
  df.drop(['Period'], axis=1, inplace=True)

  # Now filter for the 2002-2014 data and aggregate by day
  df = df.loc[df.Date < pd.to_datetime('2015-01-01'), :]
  
  return df
  
def preprocess_temperature_df(df):
  # Convert the date index to an actual date
  df['Date'] = pd.to_datetime('1899-12-30') + pd.Timedelta('1 day') * df['Date']
  df = df.loc[df.Period==30, :]
  df.drop(['Period'], axis=1, inplace=True)
  
  return df