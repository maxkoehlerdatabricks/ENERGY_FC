# Databricks notebook source
from ml_energy_forecasting.prep import data_access
from ml_energy_forecasting.prep import weather_forecast_model
from ml_energy_forecasting.utils import conf

# COMMAND ----------

data_access.load_and_overwrite(
  database_name=conf.DATABASE_NAME, delta_location=conf.DELTA_LOCATION
)

weather_forecast_model.add_model_to_registry(
  model_name=conf.WEATHER_MODEL_NAME
)
