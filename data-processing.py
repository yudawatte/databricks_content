# Databricks notebook source
dbutils.widgets.dropdown("exclude_pms", "False", ["True", "False"], "Exclude PMS")
dbutils.widgets.dropdown("target_type", "REVENUE", ["REVENUE", "ROOMS"], "Target Type")
dbutils.widgets.text("selected_hotels", "","Hotels")
dbutils.widgets.text("selected_metrics", "","Metrics")
dbutils.widgets.dropdown("env_stage", "dev", ["dev", "prod"], "Pipeline Stage")
dbutils.widgets.text("prediction_horizon", "28", "Prediction Horizon")
dbutils.widgets.text("lead_window", "100", "Lead Window")
dbutils.widgets.text("partition_date", "2022-08-01", "Partition Date")
dbutils.widgets.text("model_start_date", "2018-10-01", "Model Start Date")
dbutils.widgets.text("eval_date", "2023-03-03", "Evaluation End Date")

# COMMAND ----------

# MAGIC %pip install mlflow==2.2.2 --quiet

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
from pyspark.sql.functions import *
from pyspark.sql.types import StringType, DateType, IntegerType, StructField, StructType, DoubleType, LongType
from sktime.transformations.series.date import DateTimeFeatures
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error, MeanAbsolutePercentageError, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

import holidays
import matplotlib
import seaborn as sns
import datetime
from pathlib import Path
import pickle
import os
import sys

from mapie.regression import MapieRegressor
from mapie.quantile_regression import MapieQuantileRegressor
from sys import version_info

import cloudpickle
import mlflow
import mlflow.pyfunc
from autogluon.tabular import TabularDataset, TabularPredictor
from pyspark.sql.functions import pandas_udf, PandasUDFType

import logging
import warnings
import mlflow
from mlflow import MlflowException
import time

warnings.filterwarnings("ignore")
start_time = time.perf_counter()

# COMMAND ----------

# Util functions
def extract_param_values(value):
    if value == "":
        return []
    elif "," in value:
        val_lst = value.split(",")
        return val_lst
    else:
        return [value]

def str_to_bool(value):
    FALSE_VALUES = ['false', 'no', '0']
    TRUE_VALUES = ['true', 'yes', '1']
    lvalue = str(value).lower()
    if lvalue in (FALSE_VALUES): return False
    if lvalue in (TRUE_VALUES):  return True
    raise Exception("String value should be one of {}, but got '{}'.".format(FALSE_VALUES + TRUE_VALUES, value))

# COMMAND ----------

# Read params
params = {}
params['ENV'] = getArgument("env_stage")
params['WITHOUT_PMS'] = str_to_bool(getArgument("exclude_pms"))
params['TARGET_TYPE'] = getArgument("target_type")
params['SELECTED_HOTELS'] = extract_param_values(getArgument("selected_hotels"))
params['SELECTED_METRICS'] = [int(x) for x in extract_param_values(getArgument("selected_metrics"))]
params['PREDICTION_HORIZON'] = int(getArgument("prediction_horizon")) 
params['LEAD_WINDOW'] = int(getArgument("lead_window")) 
params['PARTITION_DATE'] = pd.to_datetime(getArgument("partition_date"))
params['MODEL_START_DATE'] = pd.to_datetime(getArgument("model_start_date"))
params['COVID_START_DATE'] = pd.to_datetime('2020-03-01')
params['COVID_END_DATE'] = pd.to_datetime('2021-08-01')
params['EVAL_END_DAY'] = pd.to_datetime(getArgument("eval_date"))
params['REVENUE_COL'] = "_reservationRevenuePerRoomUSD"
params['ROOMS_COL'] = "_rooms"
params['CALC_UNCERTAINTY'] = False
params['MODEL_TYPE'] = "AG"
params['LAG_NUMBERS'] = [1,7,14,21,28]

if params['TARGET_TYPE'] == "ROOMS":
    params['TARGET_COLUMN'] = params['ROOMS_COL']    
elif params['TARGET_TYPE'] == "REVENUE":
    params['TARGET_COLUMN'] = params['REVENUE_COL']

params['HIVE_TABLE_NAME'] = f"global_forecast_preprocessed"
params['LOG_ROOT'] = '/dbfs/mnt/extractionlogs/synxis'
params['REPOPATH'] = "/Workspace/Repos/manik@surge.global/phg-data-mlsys/src"
#params['REPOPATH'] = "/Workspace/Repos/yasith.udawatte@henrymwuamica.onmicrosoft.com/phg-data-mlsys/src"

params['ML_EXPERIMENT_ID'] = 2947576836153301
params['SAVE_MODEL']= True
params['SAVE_METRICS'] = True

# COMMAND ----------

cluster_name = spark.conf.get("spark.databricks.clusterUsageTags.clusterName")

if (params['ENV'] == "dev") and ("dev" in cluster_name):
    print(f"Loading phgml package from repo {params['REPOPATH']}")
    sys.path.append(os.path.abspath(params['REPOPATH']))

# COMMAND ----------

from phgml.data.processing_distr import calc_date_features, add_date_features , preprocess_data, filter_data, aggregate_target, create_rows, compile_train_table
from phgml.data.processing import get_lags
from phgml.reporting.visualization import *
from phgml.data.data_types import revenue_preprocessed_schema
from phgml.reporting.logging import get_logging_path, get_logging_filename

# COMMAND ----------

"""
Hotels being considered for this experiment :

Downtown Grand Hotel & Casino - 10443
Hotel Californian - 71999
Pendry West Hollywood - 9718
Montage Laguna Beach - 26532
Boston Harbor Hotel	- 26834
NoMo SoHo - 64942
The Dominick - 1406
Balboa Bay Resort - 63662
Windsor Court Hotel	- 55810

"""

# COMMAND ----------

print(f"Using training data up to {params['PARTITION_DATE']}")

# COMMAND ----------

for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
logging.root.setLevel(logging.INFO)

processing_timestamp = datetime.datetime.now()

logfile_path = get_logging_path(params['LOG_ROOT'],processing_timestamp)
if not os.path.exists(logfile_path):
    os.makedirs(logfile_path)

pms = "PMS"
if params['WITHOUT_PMS']:
    pms = "NOPMS"
        
log_file_name = get_logging_filename(
    logfile_path,
    "PREPROCESS",
    params['TARGET_TYPE'],
    pms,
    processing_timestamp)

logger = logging.getLogger(f"preprocess-{params['TARGET_TYPE']}-{pms}")

file_handler = logging.FileHandler(log_file_name)
file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_format)

logger.addHandler(file_handler)

# COMMAND ----------

logger.info("Selecting hotels.")
hotel_details = spark.sql("select distinct HotelID,HotelName from phg_data.dim_hotels_").toPandas()
hotel_ids = hotel_details["HotelID"].values

correct_hotel_ids = []
unknown_hotel_ids = []

for selected_hotel in params['SELECTED_HOTELS']:
    if selected_hotel in hotel_ids:
        correct_hotel_ids.append(selected_hotel)
        logger.info(f"Hotel ID {params['SELECTED_HOTELS']} will be selected.")
    else:
        logger.info(f"Unknown hotel {params['SELECTED_HOTELS']} provided at hotel selection.")
        unknown_hotel_ids.append(selected_hotel)

if unknown_hotel_ids:
    logger.error(f"Following unknown hotel ids were provided at hotel selection: {unknown_hotel_ids}")
    raise ValueError(f"Following unknown hotel ids were provided at hotel selection: {unknown_hotel_ids}")

params['SELECTED_HOTELS'] = correct_hotel_ids

# COMMAND ----------

logger.info("Loading data")
columns = ["HotelID","_StayDates","confirmationDate","departureDate","channel","status",params['REVENUE_COL'],params['ROOMS_COL']]
dfsp =  spark.sql(f"select {','.join(columns)} from phg_data.consumption_deaggrecords where status='Confirmed' ")

if correct_hotel_ids:
    dfsp = dfsp.filter(dfsp.HotelID.isin(correct_hotel_ids))

logger.info("Preprocessing data")
df = preprocess_data(dfsp,
                    params['WITHOUT_PMS'],
                    params['REVENUE_COL'],
                    params['ROOMS_COL'],
                    params['MODEL_START_DATE'],
                    params['COVID_START_DATE'],
                    params['COVID_END_DATE'])

dates = calc_date_features(df)

df_lags = get_lags(df.toPandas(),lag_numbers=params['LAG_NUMBERS'],target_col=params['TARGET_COLUMN'])

logger.info(f"Stay dates filtering upto : {params['PARTITION_DATE']}")
df = filter_data(df=df,
                 partition=params['PARTITION_DATE'],
                 revenue_col=params['REVENUE_COL'],
                 rooms_col=params['ROOMS_COL'])
                 
logger.info(f"Processing data for target type: {params['TARGET_TYPE'] } : {params['TARGET_COLUMN']}")
logger.info(f"Excluding PMS data? {params['WITHOUT_PMS']}")
logger.info("Compiling test data set")

output_df  = compile_train_table(
    df, 
    df_lags, 
    dates, 
    target_column=params['TARGET_COLUMN'],
    booking_lead_end=params['LEAD_WINDOW'])

# COMMAND ----------

file_format = "delta"

output_df = spark.createDataFrame(output_df)

logger.info(f"Writing preprocessed data to table {params['HIVE_TABLE_NAME']}")
(output_df.write
         .format("delta")
         .mode("overwrite")
         .partitionBy("HotelID")
         .option("overwriteSchema", "true")
         .saveAsTable(params['HIVE_TABLE_NAME']))

# COMMAND ----------

dbutils.jobs.taskValues.set(key= 'ENV', value = params['ENV'])
dbutils.jobs.taskValues.set(key= 'WITHOUT_PMS', value = params['WITHOUT_PMS'])
dbutils.jobs.taskValues.set(key= 'TARGET_TYPE', value = params['TARGET_TYPE'])
dbutils.jobs.taskValues.set(key= 'SELECTED_HOTELS', value = params['SELECTED_HOTELS'])
dbutils.jobs.taskValues.set(key= 'SELECTED_METRICS', value = params['SELECTED_METRICS'])
dbutils.jobs.taskValues.set(key= 'PREDICTION_HORIZON', value = params['PREDICTION_HORIZON'])
dbutils.jobs.taskValues.set(key= 'LEAD_WINDOW', value = params['LEAD_WINDOW'])
dbutils.jobs.taskValues.set(key= 'LAG_NUMBERS', value = params['LAG_NUMBERS'])
dbutils.jobs.taskValues.set(key= 'PARTITION_DATE', value = str(params['PARTITION_DATE']))
dbutils.jobs.taskValues.set(key= 'MODEL_START_DATE', value = str(params['MODEL_START_DATE']))
dbutils.jobs.taskValues.set(key= 'COVID_START_DATE', value = str(params['COVID_START_DATE']))
dbutils.jobs.taskValues.set(key= 'COVID_END_DATE', value = str(params['COVID_END_DATE']))
dbutils.jobs.taskValues.set(key= 'EVAL_END_DAY', value = str(params['EVAL_END_DAY']))
dbutils.jobs.taskValues.set(key= 'REVENUE_COL', value = params['REVENUE_COL'])
dbutils.jobs.taskValues.set(key= 'ROOMS_COL', value = params['ROOMS_COL'])
dbutils.jobs.taskValues.set(key= 'CALC_UNCERTAINTY', value = params['CALC_UNCERTAINTY'])
dbutils.jobs.taskValues.set(key= 'MODEL_TYPE', value = params['MODEL_TYPE'])
dbutils.jobs.taskValues.set(key= 'TARGET_COLUMN', value = params['TARGET_COLUMN'])
dbutils.jobs.taskValues.set(key= 'HIVE_TABLE_NAME', value = params['HIVE_TABLE_NAME'])
dbutils.jobs.taskValues.set(key= 'LOG_ROOT', value = params['LOG_ROOT'])
dbutils.jobs.taskValues.set(key= 'REPOPATH', value = params['REPOPATH'])
dbutils.jobs.taskValues.set(key= 'ML_EXPERIMENT_ID', value = params['ML_EXPERIMENT_ID'])
dbutils.jobs.taskValues.set(key= 'SAVE_MODEL', value = params['SAVE_MODEL'])
dbutils.jobs.taskValues.set(key= 'SAVE_METRICS', value = params['SAVE_METRICS'])

# COMMAND ----------

elapsed_time = time.perf_counter() - start_time
logger.info(f"Time elapsed {elapsed_time}")
logger.info(f"Time elapsed in minutes {elapsed_time/60}")
logger.info("Preprocessing completed.")