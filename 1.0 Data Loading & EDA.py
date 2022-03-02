# Databricks notebook source
# MAGIC %md 
# MAGIC 
# MAGIC # 1. Visualising & Preprocessing

# COMMAND ----------

from ml_energy_forecasting.utils import conf
sql(f'use {conf.DATABASE_NAME}')

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from demand

# COMMAND ----------

# DBTITLE 1,Run any SQL & Visualise data
# MAGIC %sql
# MAGIC select * from demand
# MAGIC where date > '2012-01-01'
# MAGIC order by date

# COMMAND ----------

# DBTITLE 1,Create Spark SQL temporary view
# MAGIC %sql
# MAGIC drop table if exists tmp_energy_forecasting_features;
# MAGIC 
# MAGIC create temporary view tmp_energy_forecasting_features as 
# MAGIC   select f.Date, OperationalLessIndustrial, Industrial, Temp, case when isHoliday != 0 then 1 else 0 end as isHoliday
# MAGIC   from demand f
# MAGIC     join temperature t on f.Date = t.Date
# MAGIC     left join holiday h on f.Date = to_date(h.Date, 'MM/dd/yyyy')

# COMMAND ----------

spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")

# COMMAND ----------

# MAGIC %sql
# MAGIC drop table if exists tmp_energy_forecasting_features_agg;
# MAGIC 
# MAGIC create temporary view tmp_energy_forecasting_features_agg as 
# MAGIC   select
# MAGIC     date, sum(operationalLessIndustrial) as energy_consumption, first(isHoliday) as is_holiday, avg(temp) as avg_temp 
# MAGIC   from tmp_energy_forecasting_features
# MAGIC   group by date;
# MAGIC   
# MAGIC select * from tmp_energy_forecasting_features_agg

# COMMAND ----------

# MAGIC %md #2. Moving to Python

# COMMAND ----------

import datetime
import pandas as pd

# COMMAND ----------

dbutils.data.summarize(table('tmp_energy_forecasting_features_agg'))

# COMMAND ----------

# DBTITLE 1,We can write Pandas code for more complex transformations
df = spark.sql('select * from tmp_energy_forecasting_features_agg').toPandas()
display(df)

# COMMAND ----------

def get_day_and_month(timestamp):
  day = timestamp.date().day
  month = timestamp.date().month
  
  out = datetime.datetime.strptime('{}-{}'.format(month, day), '%m-%d')
  return out


df['year'] = df['date'].apply(lambda x: x.year)
df['month'] = df['date'].apply(lambda x: x.month)
df['day'] = df['date'].apply(lambda x: x.day)

df['y'] = df['energy_consumption']

year_dfs = [df for _, df in df.groupby('year')]

year_df = pd.concat([
  df \
    .set_index(df.apply(lambda row: '{}_{}'.format(row['month'], row['day']), axis=1))[['y', 'avg_temp']] \
    .rename(columns={'y': 'y_{}'.format(df['year'].iloc[0]), 'avg_temp': 'avg_temp_{}'.format(df['year'].iloc[0])})
  for df in year_dfs
], axis=1, sort=True)

year_df['month'] = year_df.index.map(lambda x: int(x.split('_')[0]))
year_df['day'] = year_df.index.map(lambda x: int(x.split('_')[1]))

year_df = year_df.fillna(method='ffill')

# COMMAND ----------

display(year_df)

# COMMAND ----------

# DBTITLE 1,Save our aggregated data to delta
df = spark.sql('select * from tmp_energy_forecasting_features_agg order by date asc').toPandas()
df['ds'] = df['date'] + (datetime.datetime.now() - df['date'].max()) # this is just to simulate current data, we are "shifting" it
df['y'] = df['energy_consumption']
df['avg_temp_squared'] = df['avg_temp'] * df['avg_temp']

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md #3. Delta for Data Science & Feature Store

# COMMAND ----------

f'{conf.DATABASE_NAME}.energy_consumption_daily'

# COMMAND ----------

from databricks import feature_store
fs = feature_store.FeatureStoreClient()

fs.create_table(
    name=f'{conf.DATABASE_NAME}.energy_consumption_daily',
    primary_keys=['date'],
    df=spark.createDataFrame(df),
    description='Daily aggregate statistics of energy consumption in Victoria',
)

# COMMAND ----------

# MAGIC %sql 
# MAGIC --DROP TABLE IF EXISTS energy_forecasting__max_kohler.energy_consumption_daily;

# COMMAND ----------

# MAGIC %sql describe energy_consumption_daily

# COMMAND ----------

# MAGIC %sql describe history energy_consumption_daily

# COMMAND ----------


