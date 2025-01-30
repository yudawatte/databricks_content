# Databricks notebook source
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
import os
import sys
from sys import version_info
import cloudpickle
import matplotlib.ticker as ticker
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error, MeanAbsolutePercentageError, mean_absolute_error

# COMMAND ----------

params = {}
params['ENV'] = dbutils.jobs.taskValues.get(taskKey = "data-preprocessing", key = "ENV")
params['WITHOUT_PMS'] = dbutils.jobs.taskValues.get(taskKey = "data-preprocessing", key = "WITHOUT_PMS")
params['TARGET_TYPE'] = dbutils.jobs.taskValues.get(taskKey = "data-preprocessing", key = "TARGET_TYPE")
params['SELECTED_HOTELS'] = dbutils.jobs.taskValues.get(taskKey = "data-preprocessing", key = "SELECTED_HOTELS")
params['SELECTED_METRICS'] = dbutils.jobs.taskValues.get(taskKey = "data-preprocessing", key = "SELECTED_METRICS")
params['PREDICTION_HORIZON'] = dbutils.jobs.taskValues.get(taskKey = "data-preprocessing", key = "PREDICTION_HORIZON")
params['LEAD_WINDOW'] = dbutils.jobs.taskValues.get(taskKey = "data-preprocessing", key = "LEAD_WINDOW")
params['LAG_NUMBERS'] = dbutils.jobs.taskValues.get(taskKey = "data-preprocessing", key = "LAG_NUMBERS")
params['PARTITION_DATE'] = pd.to_datetime(dbutils.jobs.taskValues.get(taskKey = "data-preprocessing", key = "PARTITION_DATE"))
params['MODEL_START_DATE'] = pd.to_datetime(dbutils.jobs.taskValues.get(taskKey = "data-preprocessing", key = "MODEL_START_DATE"))
params['COVID_START_DATE'] = pd.to_datetime(dbutils.jobs.taskValues.get(taskKey = "data-preprocessing", key = "COVID_START_DATE"))
params['COVID_END_DATE'] = pd.to_datetime(dbutils.jobs.taskValues.get(taskKey = "data-preprocessing", key = "COVID_END_DATE"))
params['EVAL_END_DAY'] = pd.to_datetime(dbutils.jobs.taskValues.get(taskKey = "data-preprocessing", key = "EVAL_END_DAY"))
params['REVENUE_COL'] = dbutils.jobs.taskValues.get(taskKey = "data-preprocessing", key = "REVENUE_COL")
params['ROOMS_COL'] = dbutils.jobs.taskValues.get(taskKey = "data-preprocessing", key = "ROOMS_COL")
params['CALC_UNCERTAINTY'] = dbutils.jobs.taskValues.get(taskKey = "data-preprocessing", key = "CALC_UNCERTAINTY")
params['MODEL_TYPE'] = dbutils.jobs.taskValues.get(taskKey = "data-preprocessing", key = "MODEL_TYPE")
params['TARGET_COLUMN'] = dbutils.jobs.taskValues.get(taskKey = "data-preprocessing", key = "TARGET_COLUMN")
params['HIVE_TABLE_NAME'] = dbutils.jobs.taskValues.get(taskKey = "data-preprocessing", key = "HIVE_TABLE_NAME")
params['LOG_ROOT'] = dbutils.jobs.taskValues.get(taskKey = "data-preprocessing", key = "LOG_ROOT")
params['REPOPATH'] = dbutils.jobs.taskValues.get(taskKey = "data-preprocessing", key = "REPOPATH")
params['ML_EXPERIMENT_ID'] = dbutils.jobs.taskValues.get(taskKey = "data-preprocessing", key = "ML_EXPERIMENT_ID")
params['SAVE_MODEL'] = dbutils.jobs.taskValues.get(taskKey = "data-preprocessing", key = "SAVE_MODEL")
params['SAVE_METRICS'] = dbutils.jobs.taskValues.get(taskKey = "data-preprocessing", key = "SAVE_METRICS")

#params['RUN_ID'] = dbutils.jobs.taskValues.get(taskKey = "forecast-training", key = "RUN_ID")
params['SLIDE_STEP'] = dbutils.jobs.taskValues.get(taskKey = "sliding-inference", key = "SLIDE_STEP")
params['SLIDING_RESULTS_TABLE'] = dbutils.jobs.taskValues.get(taskKey = "sliding-inference", key = "SLIDING_RESULTS_TABLE")

# COMMAND ----------

import warnings
warnings.filterwarnings("ignore")

# COMMAND ----------

cluster_name = spark.conf.get("spark.databricks.clusterUsageTags.clusterName")

if (params['ENV'] == "dev") and ("dev" in cluster_name):
    print(f"Loading phgml package from repo {params['REPOPATH']}")
    sys.path.append(os.path.abspath(params['REPOPATH']))

# COMMAND ----------

# Setting config for the seaborn plots
sns.set_style("darkgrid")
pd.plotting.register_matplotlib_converters()
# Default figure size
sns.mpl.rc("figure", figsize=(16, 6))
sns.mpl.rc("font", size=14)

pd.set_option('display.max_columns', None)

# COMMAND ----------

hotel_ids = spark.sql("select distinct HotelID,HotelName from phg_data.consumption_deaggrecords").toPandas()
hotel_ids = hotel_ids[hotel_ids["HotelID"].isin(params['SELECTED_HOTELS'])]
hotel_ids = hotel_ids.sort_values("HotelID")
hotel_ids["HotelName"] = hotel_ids["HotelName"].apply(lambda x:x[:20])

# COMMAND ----------

df = spark.sql(f"select * from {params['SLIDING_RESULTS_TABLE']}").toPandas()
df = df[df["HotelID"].isin(params['SELECTED_HOTELS'])]

# COMMAND ----------

smape_lst = []
mae_lst = []
metric_df_lst = []
for n,group in df.groupby(["HotelID","_StayDates","eval_start","eval_end"]):
    temp_dic = {}
    temp_dic['HotelID'] = [n[0]]
    temp_dic['_StayDates'] = [n[1]]
    temp_dic['eval_start'] = [n[2]]
    temp_dic['eval_end'] = [n[3]]
    for metric in params['SELECTED_METRICS']:
        smape = mean_absolute_percentage_error(group["y_pred"][:metric],group["y_true"][:metric],symmetric=True)
        mae = mean_absolute_error(group["y_pred"][:metric],group["y_true"][:metric])
    
        temp_dic[f'SMAPE{metric}'] = smape
        temp_dic[f'MAE{metric}'] = mae
    
    dfpart = pd.DataFrame(temp_dic)
    metric_df_lst.append(dfpart)

# COMMAND ----------

metrics = pd.concat(metric_df_lst,axis=0)
metrics = metrics.merge(hotel_ids,left_on="HotelID",right_on="HotelID")

# COMMAND ----------

metrics.groupby(["HotelID","HotelName"]).mean()

# COMMAND ----------

for metric in params['SELECTED_METRICS']:
    plt.figure(figsize=(9,5))
    plt.title(f"SMAPE {metric} days")
    met_col = f"SMAPE{metric}"
    ax = sns.barplot(y="HotelName", x=met_col, data=metrics, estimator=np.mean, ci=95, capsize=.2, color='lightblue')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.tick_params(axis='x', rotation=90)
    ax.set(xlim=(0,1))


# COMMAND ----------

for metric in params['SELECTED_METRICS']:
    plt.figure(figsize=(9,5))
    plt.title(f"MAE {metric} days")
    met_col = f"MAE{metric}"
    ax = sns.barplot(y="HotelName", x=met_col, data=metrics, estimator=np.mean, ci=95, capsize=.2, color='lightblue')
    ax.tick_params(axis='x', rotation=90)
    ax.set(xlim=(0, 25000))

# COMMAND ----------

metrics.groupby(["eval_start","eval_end","HotelName"]).mean()

# COMMAND ----------

metrics.groupby(["eval_start","eval_end"]).mean()