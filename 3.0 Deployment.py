# Databricks notebook source
import logging
import warnings

warnings.simplefilter("ignore", UserWarning)
logging.getLogger("py4j").setLevel(logging.ERROR) 

# COMMAND ----------

import numpy as np
import pandas as pd

import mlflow
import mlflow.pyfunc
import mlflow.sklearn

from ml_energy_forecasting.utils import conf

# COMMAND ----------

client = mlflow.tracking.MlflowClient()

# COMMAND ----------

# MAGIC %md ### Loading energy forecast mode

# COMMAND ----------

model_name = f'energy-forecast-prophet__{conf.USERNAME_FORMATTED}'
model = mlflow.pyfunc.load_model(f'models:/{model_name}/Production')

# COMMAND ----------

# MAGIC %md ### ...and realising that we need weather data to use it

# COMMAND ----------

weather_model_name = conf.WEATHER_MODEL_NAME
weather_model = mlflow.sklearn.load_model(f'models:/{weather_model_name}/Production')

# COMMAND ----------

weather_forecast = weather_model.predict(15, 'Victoria')
weather_forecast['is_holiday'] = weather_forecast['is_holiday'].apply(np.int32)
display(weather_forecast)

# COMMAND ----------

# MAGIC %md ### Now that we have weather data, we can predict Energy Consumption

# COMMAND ----------

weather_forecast = weather_forecast.astype({'is_holiday': 'int32'})
df_forecast = model.predict(weather_forecast.rename(columns={'time': 'date', 'avg': 'avg_temp', 'avg_squared': 'avg_temp_squared'}))

# COMMAND ----------

df = spark.sql(f'select * from {conf.DATABASE_NAME}.energy_consumption_daily order by ds asc').toPandas()

# COMMAND ----------

df_all = pd.concat([df, df_forecast[['date', 'yhat', 'yhat_lower', 'yhat_upper']]]).reset_index()
df_all['date'] = df_all['date'].apply(str)
display(df_all[-100:])

# COMMAND ----------

# MAGIC %md ### Now let's pretend I push it out to a Database / Dashboard. I can now schedule this to repeat the process automatically.
