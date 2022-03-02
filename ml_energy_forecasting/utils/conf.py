import os

def get_username():
  path = os.getcwd()
  username = path.split('/')[3]
  return username

def get_username_formatted():
  username = get_username()
  out = username.split('@')[0].replace('.', '_')
  return out


USERNAME = get_username()
USERNAME_FORMATTED = get_username_formatted()

DELTA_LOCATION = f'dbfs:/Users/{USERNAME}/energy_forecasting/'

DATABASE_NAME = f'energy_forecasting__{USERNAME_FORMATTED}'
WEATHER_MODEL_NAME = f'weather-forecast-api__{USERNAME_FORMATTED}'