import mlflow
import mlflow.sklearn

import pandas as pd

import time
import datetime
import requests


MODEL_NAME = 'weather-forecast-api'
client = mlflow.tracking.MlflowClient()


class WeatherModel(object):
  def predict(self, num_days, state='Victoria'):
    api_response = self.get_victoria_weather_forecast(num_days)
    df = self.parse_weather_json(api_response)
    return df
  
  @staticmethod
  def get_victoria_weather_forecast(num_days):
    start_day = datetime.datetime.now()
    end_day = start_day + datetime.timedelta(days=num_days-1)
      
    url = 'https://api.tomorrow.io/v4/timelines/'
    querystring = {'apikey': 'MhsPrDXLlKMj8RwH9wnvTHtCU5BJ6A5X'}
    payload = {
      'fields': ['temperature'],
      'units': 'metric',
      'timesteps': ['1d'],
      'location': '-37.851989144, 144.926337'
    }
    headers = {
      'Accept': 'application/json',
      'Content-Type': 'application/json'
    }
    response = requests.request('POST', url, json=payload, headers=headers, params=querystring).json()
    return response
  
  @staticmethod
  def parse_weather_json(json_data):
    data = [
      {'time': d['startTime'], 'avg': d['values']['temperature']}
      for d in 
      json_data['data']['timelines'][0]['intervals']
    ]
    
    df = pd.DataFrame(data)
    df['avg_squared'] = df['avg'] * df['avg']
    df['is_holiday'] = 0.0
    return df
  
  
def add_model_to_registry(model_name=MODEL_NAME):
  with mlflow.start_run():
    model = WeatherModel()
    mlflow.sklearn.log_model(model, artifact_path='model')
    run_id = mlflow.active_run().info.run_id
    
  result = mlflow.register_model(
    'runs:/{}/model'.format(run_id),
    model_name
  )

  latest_version = client.get_latest_versions(model_name, ['None'])[0].version

  # transition model we just created into Production
  client.transition_model_version_stage(
      name=model_name,
      version=latest_version,
      stage='Production'
  )
  
  
  description = """
  ***What it does:***
  Weather forecaseting "model" that is simply an API call behind the scenes.
  Can forecast weather for up to 15 days for the various locations in Australia. Output is Pandas dataframe with forecast and some metadata.

  ***Example usage:***
  ```
  model.predict(n_days=15, 'Victoria')
  ```

  yields:
  ```
  time	min	max	avg	avg_squared	is_holiday
  0	2020-05-03T18:00:00	8.36	12.44	10.400	108.160000	0.0
  1	2020-05-03T20:00:00	9.36	15.08	12.220	149.328400	0.0
  2	2020-05-04T21:00:00	9.15	17.90	13.525	182.925625	0.0
  ```
  """
  
  client.update_model_version(
    name=model_name,
    version=latest_version,
    description=description
  )
