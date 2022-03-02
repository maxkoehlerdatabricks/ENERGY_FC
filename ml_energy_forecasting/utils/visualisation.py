import matplotlib.pyplot as plt
import plotly.express as px

import datetime
import pandas as pd

# These functions build plots which we use and save as artifacts for the experiments.
def plot_forecasts(vic_df, forecast_df):
  vic_df['kind'] = 'actuals'
  forecast_df['kind'] = 'forecast'
  
  df_all = pd.concat([
    vic_df[['date', 'energy_consumption', 'kind']],
    forecast_df[['date', 'yhat', 'kind']]
  ])
  
  fig = px.line(df_all, x='date', y=['energy_consumption', 'yhat'], color='kind', title='Victoria Energy Forecast')
  return fig

def prophet_plot(vic_df, forecast_df, mdl):
  df = vic_df.merge(forecast_df[['energy_consumption', 'yhat']], on='energy_consumption',how='left')

  fig = mdl.plot(forecast_df)
  ax = fig.gca()
  a = add_changepoints_to_plot(ax, mdl, forecast_df)
  ax.set_xlim([vic_df['date'].max() - datetime.timedelta(days=365*3), vic_df['date'].max()])
  ax.set_ylabel('Demand')
  ax.set_title('Victorian Energy Consumption')
  plt.close('all')
  return fig

def components_plot(vic_df, forecast_df, mdl):
  df = vic_df.merge(forecast_df[['energy_consumption','yhat']], on='date',how='left')

  fig = mdl.plot_components(forecast_df)
  ax = fig.gca()
  ax.set_xlim([vic_df['date'].max() - datetime.timedelta(days=365*3), vic_df['date'].max()])
  ax.set_ylabel('Demand')
  ax.set_title('Victorian Energy Consumption')
  plt.close('all')
  return fig