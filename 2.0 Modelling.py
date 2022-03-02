# Databricks notebook source
import logging
import warnings

warnings.simplefilter("ignore", UserWarning)
logging.getLogger("py4j").setLevel(logging.ERROR)

# COMMAND ----------

# MAGIC %md # 1. A Little bit about MLFlow & Databricks for Data Science

# COMMAND ----------

import datetime
import pandas as pd
import mlflow

#from ml.model_run import run_experiment
from ml_energy_forecasting.utils import conf
from ml_energy_forecasting.ml.model import EnergyConsumption
from ml_energy_forecasting.ml.model_run import get_features, train_and_log_model

# COMMAND ----------

# MAGIC %md # 2. Run Prophet models
# MAGIC ![Prophet logo](https://miro.medium.com/max/964/0*tVCene42rgUTNv9Q.png)
# MAGIC 
# MAGIC Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. It works best with time series that have strong seasonal effects and several seasons of historical data. Prophet is robust to missing data and shifts in the trend, and typically handles outliers well.
# MAGIC 
# MAGIC Prophet is open source software released by Facebook’s Core Data Science team.

# COMMAND ----------

#experiment_id = 4198057744178530 # setting experiment id 
experiment_id = 470236344233460

# COMMAND ----------

# MAGIC %md ### 2.1 Baseline Model

# COMMAND ----------

import pyspark.sql.functions as F
import pandas as pd

# COMMAND ----------

run_name = 'Base Prophet Model - additive'

params = {'seasonality_mode': 'additive', 'daily_seasonality': False}
col_mapping = {'date': 'ds', 'energy_consumption': 'y'}
xreg_list = ['is_holiday']

with mlflow.start_run(run_name=run_name, experiment_id=experiment_id):
  train_df, test_df, training_set = get_features(features=xreg_list, train_perc=0.8, tbl=f'{conf.DATABASE_NAME}.energy_consumption_daily')
  result = train_and_log_model(train_df, test_df, EnergyConsumption, params, col_mapping, xreg_list, training_set)
  
display(result['plots']['forecast'])

# COMMAND ----------

run_name = 'Base Prophet Model - multiplicative'

params = {'seasonality_mode': 'multiplicative', 'daily_seasonality': False}
col_mapping = {'date': 'ds', 'energy_consumption': 'y'}
xreg_list = ['avg_temp']

with mlflow.start_run(run_name=run_name, experiment_id=experiment_id):
  train_df, test_df, training_set = get_features(features=xreg_list, train_perc=0.8, tbl=f'{conf.DATABASE_NAME}.energy_consumption_daily')
  result = train_and_log_model(train_df, test_df, EnergyConsumption, params, col_mapping, xreg_list, training_set)
  
display(result['plots']['forecast'])

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### 2.2 Model with more regressors
# MAGIC Let's now add another "features" to our model. Namely, we think that model can improve if we add:
# MAGIC - Daily average temperature forecast
# MAGIC - Holiday flag

# COMMAND ----------

run_name = 'Prophet Model - additive'

params = {'seasonality_mode': 'additive', 'daily_seasonality': False}
col_mapping = {'date': 'ds', 'energy_consumption': 'y'}
xreg_list = ['avg_temp', 'avg_temp_squared', 'is_holiday']

with mlflow.start_run(run_name=run_name, experiment_id=experiment_id):
  train_df, test_df, training_set = get_features(features=xreg_list, train_perc=0.8, tbl=f'{conf.DATABASE_NAME}.energy_consumption_daily')
  result = train_and_log_model(train_df, test_df, EnergyConsumption, params, col_mapping, xreg_list, training_set)
  
display(result['plots']['forecast'])

# COMMAND ----------

# MAGIC %md ### 2.3 Let HyperOpt find the best model
# MAGIC Use HyperOpt with Spark trials to run distributed hyperparameters tuning across workers in parallel
# MAGIC 
# MAGIC ![my_test_image](https://www.jeremyjordan.me/content/images/2017/11/grid_search.gif)
# MAGIC ![my_test_image](https://www.jeremyjordan.me/content/images/2017/11/Bayesian_optimization.gif)

# COMMAND ----------

from functools import partial
spark.conf.set("spark.databricks.mlflow.trackHyperopt.enabled", False)

# COMMAND ----------

from hyperopt import SparkTrials, Trials, hp, fmin, tpe, STATUS_FAIL, STATUS_OK

params = {
  'growth': hp.choice('growth', ['linear']),
  'changepoint_prior_scale': hp.uniform('changepoint_prior_scale', 0.1, 0.8),
  'changepoint_range': hp.uniform('changepoint_range', 0.1, 0.8)
}

xreg_list = ['avg_temp', 'avg_temp_squared', 'is_holiday']
col_mapping = {'date': 'ds', 'energy_consumption': 'y'}
train_df, test_df, training_set = get_features(features=xreg_list, train_perc=0.8, tbl=f'{conf.DATABASE_NAME}.energy_consumption_daily')

def fmin_objective(params):
  try:
    with mlflow.start_run(run_name='Prophet Model - HyperOpt', experiment_id=experiment_id):
      result = train_and_log_model(train_df, test_df, EnergyConsumption, params, col_mapping, xreg_list, training_set)
      out = {'loss': result['metrics']['mape'], 'status': STATUS_OK}
    
  except Exception as E:
    out = {'loss': 1e6, 'status': STATUS_FAIL}
  
  return out


best_param = fmin(
  fn=fmin_objective, 
  space=params, 
  algo=tpe.suggest, 
  max_evals=5, 
  trials=Trials()
) 

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Register best model

# COMMAND ----------

import pprint
pp = pprint.PrettyPrinter(indent=4)
mlflow_client = mlflow.tracking.MlflowClient()

model_name = f'energy-forecast-prophet__{conf.USERNAME_FORMATTED}'

# COMMAND ----------

# Let's get the best model we have so far
best_run = mlflow_client.search_runs(experiment_ids=[experiment_id], max_results=1, order_by=['metrics.MAPE ASC'])[0].to_dictionary()
pp.pprint(best_run)

# COMMAND ----------

mlflow.register_model('runs:/' + best_run['info']['run_id'] + '/model', model_name)
displayHTML(f"<h2>Check out the model at <a href='/#mlflow/models/{model_name}'>/#mlflow/models/{model_name}</a></h2>")

# COMMAND ----------


