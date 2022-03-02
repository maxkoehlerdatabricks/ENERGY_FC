import mlflow

import numpy as np
import pandas as pd

from databricks.feature_store import FeatureLookup, FeatureStoreClient
from pyspark.sql import SparkSession

from ml_energy_forecasting.utils.visualisation import plot_forecasts

spark = SparkSession.builder.getOrCreate()

def mape(act, pred):
  return ( np.mean(np.abs((act-pred)/act*100)))

def rmse(act, pred):
  return ( np.sqrt(np.mean((act-pred)**2)))

def get_features(features, train_perc=0.8, tbl='demo_energy_forecasting.vic_energy_consumption_daily'):
  fs = FeatureStoreClient()
  df = spark.sql(f'select * from {tbl}').select('date', 'energy_consumption')
  feature_lookups = [
    FeatureLookup(table_name=tbl, feature_name=f, lookup_key='date')
    for f in features
  ]
  
  training_set = fs.create_training_set(
    df=df, feature_lookups=feature_lookups, label='energy_consumption'
  )

  df = training_set.load_df().orderBy('date', ascending=True).toPandas()
  
  df_train = df.iloc[:int(df.shape[0] * train_perc)]
  df_test = df.iloc[int(df.shape[0] * train_perc):]
  
  return df_train, df_test, training_set


def train_and_log_model(train_df, test_df, model, params={}, col_mapping={}, xreg_list=[], training_set=None):
    # Log parameters (key-value pair)
    mlflow.log_params(params)
    mlflow.log_params(col_mapping)
    mlflow.log_param('xreg_list', str(xreg_list))

    # build model & forecast
    _model = model(col_mapping, params, xreg_list)    
    _model.fit(train_df)
    _model.log_model(training_set)
    
    forecast_df = _model._predict(test_df)
    forecast_df = forecast_df.merge(
      test_df[['date','energy_consumption']], on='date', how='left'
    )
    
    # Log a metric; metrics can be updated throughout the run
    m_mape = mape(forecast_df.energy_consumption, forecast_df.yhat)
    m_rmse = rmse(forecast_df.energy_consumption, forecast_df.yhat)
    
    mlflow.log_metric('mape', m_mape)
    mlflow.log_metric('rmse', m_rmse)
        
    df = pd.concat([train_df, test_df])
    
    # Create some plots, and add these as "artifacts" to the experiment
    forecast_plt = plot_forecasts(df, forecast_df)
    mlflow.log_figure(forecast_plt, 'visualisations/forecast.html')

    out = {
      'metrics': {'mape': m_mape, 'rmse': m_rmse},
      'plots': {'forecast': forecast_plt},
      'data': {'df': df, 'forecast_df': forecast_df}
    } 
    
    return out