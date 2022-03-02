import pandas as pd
from pyspark.sql import SparkSession

from ml_energy_forecasting.utils import preprocessing

spark = SparkSession.builder.getOrCreate()


ORIG_DATA_LOCATION = {
  'demand': 'https://raw.githubusercontent.com/tidyverts/tsibbledata/master/data-raw/vic_elec/VIC2015/demand.csv',
  'temp': 'https://raw.githubusercontent.com/tidyverts/tsibbledata/master/data-raw/vic_elec/VIC2015/temperature.csv',
  'holiday': 'https://raw.githubusercontent.com/tidyverts/tsibbledata/master/data-raw/vic_elec/VIC2015/holidays.txt'
}

def load_and_overwrite(database_name, delta_location, orig_data_location=ORIG_DATA_LOCATION):
  df_demand = preprocessing.preprocess_demand_df(pd.read_csv(ORIG_DATA_LOCATION['demand']))
  df_temperature = preprocessing.preprocess_temperature_df(pd.read_csv(ORIG_DATA_LOCATION['temp']))
  df_holiday = pd.read_csv(ORIG_DATA_LOCATION['holiday'], names=['Date'])
  df_holiday['isHoliday'] = 1

  spark.createDataFrame(df_demand).createOrReplaceTempView('demand_tmp')
  spark.createDataFrame(df_temperature).createOrReplaceTempView('temperature_tmp')
  spark.createDataFrame(df_holiday).createOrReplaceTempView('holiday_tmp')

  spark.sql(f'drop database if exists {database_name} cascade')
  spark.sql(f'create database {database_name} location "{delta_location}"')
  spark.sql(f'create table {database_name}.demand using delta as select * from demand_tmp')
  spark.sql(f'create table {database_name}.temperature using delta as select * from temperature_tmp')
  spark.sql(f'create table {database_name}.holiday using delta as select * from holiday_tmp')
