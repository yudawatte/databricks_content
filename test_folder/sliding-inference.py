# Databricks notebook source
# MAGIC %pip install mlflow==2.2.2 --quiet
# MAGIC #%pip install autogluon.tabular[catboost]==0.5.2 --quiet
# MAGIC #%pip install autogluon.tabular[fastai]==0.5.2 --quiet

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
from pyspark.sql.functions import *
from pyspark.sql.types import *

from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import make_reduction, TransformedTargetForecaster, ForecastingPipeline
from sklearn.pipeline import make_pipeline
from sktime.transformations.series.date import DateTimeFeatures
from sktime.transformations.series.adapt import TabularToSeriesAdaptor
from sklearn.preprocessing import MinMaxScaler
from sktime.transformations.series.detrend import Detrender
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.forecasting.conformal import ConformalIntervals
from sktime.forecasting.sarimax import SARIMAX
from sktime.forecasting.arima import AutoARIMA

from sktime.performance_metrics.forecasting import mean_absolute_percentage_error, MeanAbsolutePercentageError, mean_absolute_error
from sktime.utils.plotting import plot_series, plot_correlations
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from typing import List
from sktime.forecasting.model_selection import ExpandingWindowSplitter
import holidays
import matplotlib
import seaborn as sns
import datetime
from pathlib import Path
import pickle
import os
import logging
import shutil

from mapie.regression import MapieRegressor
from mapie.quantile_regression import MapieQuantileRegressor
from sys import version_info
import mlflow.pyfunc
import cloudpickle
from autogluon.core.utils.loaders import load_pkl
import mlflow
from mlflow import MlflowException
import warnings
warnings.filterwarnings("ignore")

# COMMAND ----------

params = {}
"""params['ENV'] = "dev" #dbutils.jobs.taskValues.get(taskKey = "data-preprocessing", key = "ENV")
params['WITHOUT_PMS'] = False #dbutils.jobs.taskValues.get(taskKey = "data-preprocessing", key = "WITHOUT_PMS")
params['TARGET_TYPE'] = "ROOMS" #dbutils.jobs.taskValues.get(taskKey = "data-preprocessing", key = "TARGET_TYPE")
params['SELECTED_HOTELS'] = ['63662'] #dbutils.jobs.taskValues.get(taskKey = "data-preprocessing", key = "SELECTED_HOTELS")
params['SELECTED_METRICS'] = [28] #dbutils.jobs.taskValues.get(taskKey = "data-preprocessing", key = "SELECTED_METRICS")
params['PREDICTION_HORIZON'] = 28 #dbutils.jobs.taskValues.get(taskKey = "data-preprocessing", key = "PREDICTION_HORIZON")
params['LEAD_WINDOW'] = 100 #dbutils.jobs.taskValues.get(taskKey = "data-preprocessing", key = "LEAD_WINDOW")
params['LAG_NUMBERS'] = [1,7,14,21,28] #dbutils.jobs.taskValues.get(taskKey = "data-preprocessing", key = "LAG_NUMBERS")
params['PARTITION_DATE'] = pd.to_datetime('2022-12-31') #pd.to_datetime(dbutils.jobs.taskValues.get(taskKey = "data-preprocessing", key = "PARTITION_DATE"))
params['MODEL_START_DATE'] = pd.to_datetime('2021-01-01') #pd.to_datetime(dbutils.jobs.taskValues.get(taskKey = "data-preprocessing", key = "MODEL_START_DATE"))
params['COVID_START_DATE'] = pd.to_datetime('2020-03-01') #pd.to_datetime(dbutils.jobs.taskValues.get(taskKey = "data-preprocessing", key = "COVID_START_DATE"))
params['COVID_END_DATE'] = pd.to_datetime('2021-08-01') #pd.to_datetime(dbutils.jobs.taskValues.get(taskKey = "data-preprocessing", key = "COVID_END_DATE"))
params['EVAL_END_DAY'] = pd.to_datetime('2023-06-30') #pd.to_datetime(dbutils.jobs.taskValues.get(taskKey = "data-preprocessing", key = "EVAL_END_DAY"))
params['REVENUE_COL'] = "_reservationRevenuePerRoomUSD" #dbutils.jobs.taskValues.get(taskKey = "data-preprocessing", key = "REVENUE_COL")
params['ROOMS_COL'] = "_rooms" #dbutils.jobs.taskValues.get(taskKey = "data-preprocessing", key = "ROOMS_COL")
params['CALC_UNCERTAINTY'] = False #dbutils.jobs.taskValues.get(taskKey = "data-preprocessing", key = "CALC_UNCERTAINTY")
params['MODEL_TYPE'] = "AG" #dbutils.jobs.taskValues.get(taskKey = "data-preprocessing", key = "MODEL_TYPE")
params['TARGET_COLUMN'] = "_rooms" #dbutils.jobs.taskValues.get(taskKey = "data-preprocessing", key = "TARGET_COLUMN")
params['HIVE_TABLE_NAME'] = f"global_forecast_preprocessed" #-dbutils.jobs.taskValues.get(taskKey = "data-preprocessing", key = "HIVE_TABLE_NAME")
params['LOG_ROOT'] = '/dbfs/mnt/extractionlogs/synxis' #dbutils.jobs.taskValues.get(taskKey = "data-preprocessing", key = "LOG_ROOT")
params['REPOPATH'] = "/Workspace/Repos/manik@surge.global/phg-data-mlsys/src" #dbutils.jobs.taskValues.get(taskKey = "data-preprocessing", key = "REPOPATH")
params['ML_EXPERIMENT_ID'] = 2947576836153301 #dbutils.jobs.taskValues.get(taskKey = "data-preprocessing", key = "ML_EXPERIMENT_ID")
params['SAVE_MODEL'] = True #dbutils.jobs.taskValues.get(taskKey = "data-preprocessing", key = "SAVE_MODEL")
params['SAVE_METRICS'] = True #dbutils.jobs.taskValues.get(taskKey = "data-preprocessing", key = "SAVE_METRICS")"""

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

#params['RUN_ID'] = dbutils.jobs.taskValues.get(taskKey = "forecast_training", key = "RUN_ID")
params['RUN_ID'] = '1'
params['SLIDE_STEP'] = "1"
params['SLIDING_RESULTS_TABLE'] = "sliding_results_v3"

# COMMAND ----------

params

# COMMAND ----------

cluster_name = spark.conf.get("spark.databricks.clusterUsageTags.clusterName")

if (params['ENV'] == "dev") and ("dev" in cluster_name):
    print(f"Loading phgml package from repo {params['REPOPATH']}")
    sys.path.append(os.path.abspath(params['REPOPATH']))

# COMMAND ----------

from phgml.models.xgboost_model import XGBMultiStepPredictor
from phgml.models.autogluon_model import AutoGluonModel , AGMlflowModel
from phgml.data.processing_distr import calc_date_features, add_date_features , preprocess_data, filter_data, aggregate_target, create_rows, compile_test_table, filter_test_data, filter_test_partition 
from phgml.data.processing import get_lags
from phgml.reporting.visualization import *
from phgml.reporting.output_metrics import *
from phgml.reporting.logging import get_logging_path, get_logging_filename
from phgml.reporting.report_results import get_output_df, correct_prediction_list

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

print(f"Predictions will be made from {params['PARTITION_DATE']} to {params['EVAL_END_DAY']}")
print(f"Excluding PMS data? {params['WITHOUT_PMS']}")

# COMMAND ----------

columns = ["HotelID","_StayDates","confirmationDate","departureDate","channel","status",params['REVENUE_COL'],params['ROOMS_COL']]
df = spark.sql(f"select {','.join(columns)} from phg_data.consumption_deaggrecords where status='Confirmed' and HotelID IN ({','.join(params['SELECTED_HOTELS'])})")
df = preprocess_data(
    df,
    params['WITHOUT_PMS'],
    params['REVENUE_COL'],
    params['ROOMS_COL'],
    params['MODEL_START_DATE'],
    params['COVID_START_DATE'],
    params['COVID_END_DATE'])

# COMMAND ----------

params['WITHOUT_PMS'],params['REVENUE_COL'],params['ROOMS_COL'],params['MODEL_START_DATE'],params['COVID_START_DATE'],params['COVID_END_DATE']

# COMMAND ----------

params['PREDICTION_HORIZON']

# COMMAND ----------

dates = calc_date_features(df)
df_lags = get_lags(
    df.toPandas(),
    lag_numbers=params['LAG_NUMBERS'],
    target_col=params['TARGET_COLUMN'])

eval_dates = pd.date_range(
    params['PARTITION_DATE'],
    params['EVAL_END_DAY'],
    freq=f"{params['SLIDE_STEP']}D")

outputs = []

hotel_models = {hid:False for hid in params['SELECTED_HOTELS']}  

for start_date in eval_dates:
    end_date = start_date + pd.Timedelta(params['PREDICTION_HORIZON'],"D")
    print(f"Evaluating for the date range {start_date} to {end_date}")

    data = filter_test_partition(
        df,
        start_date,
        end_date,
        revenue_col=params['REVENUE_COL'],
        rooms_col=params['ROOMS_COL'])
    
    print(data.toPandas().shape)
    if data.toPandas().shape[0] == 0:
        print(f"Not enough data was found for the date range {start_date} - {end_date}")
        continue

    data = compile_test_table(
        data,
        df_lags,
        dates,
        target_column=params['TARGET_COLUMN'],
        booking_lead_end=params['LEAD_WINDOW'])
    
    for hotel_id in params['SELECTED_HOTELS']:
        print(f"\tEvaluating hotel id {hotel_id}")
        model_version = 1
        model_stage = "Staging"
        model_name = None
        
        testdf = data[data["HotelID"]==hotel_id]
        if testdf.shape[0] == 0:
            print(f"Not enough data was found for the date range {start_date} - {end_date} for hotel {hotel_id}")

        pms="PMS"
        if params['WITHOUT_PMS']:
            pms = "NOPMS"
        
        local_root_dir = f"CLF_{params['PREDICTION_HORIZON']}DAY_AG_{hotel_id}_{params['TARGET_TYPE']}_{pms}/"

        ag_model = AutoGluonModel(
            prediction_horizon=params['PREDICTION_HORIZON'],
            calc_uncertainty=params['CALC_UNCERTAINTY'],
            mlflow_run_id=params['RUN_ID'],
            hotel_id=hotel_id,
            target_type=params['TARGET_TYPE'],
            exclude_pms=params['WITHOUT_PMS'],
            local_root_dir=local_root_dir,)
    
        ag_model.model_type = f"CLF_{params['PREDICTION_HORIZON']}DAY_AG"

        ag_model.set_latest_model_version()
        model_name = [ag_model.get_model_name() for step in range(1,params['PREDICTION_HORIZON'] + 1)]
        model_version = int(ag_model.version)
        #model_version = 1 # Testing

        print(f"Model version: {model_version}")
        
        try:
            if not hotel_models[hotel_id]:
                print("Loading model from registry")
                # Load one model version before the latest version as the latest version was trained with data after 2022-08-01
                #model = ag_model.load_pyfunc_model(tag=str(int(ag_model.version) - 1))
                
                # Always get the model version 1 to make sure we get models trained with data before 2022-08-01
                model = ag_model.load_pyfunc_model(tag=str(model_version))
                
                # model = ag_model.load_pyfunc_model(tag=ag_model.version)
                hotel_models[hotel_id] = True
            else:
                print("Loading cached model")
                model = load_pkl.load(path=local_root_dir+"artifacts/model.pkl")            
                
            y_pred , y_test, y_upper, y_lower  = model.predict(testdf)
        
        except Exception as e:
            raise e
        #finally:
        #    print("Cleaning models f")
        #    ag_model.clean()
        #    if os.path.exists(local_root_dir):
        #        shutil.rmtree(local_root_dir)
        
        dflst = []
        for i,stay_date in enumerate(testdf["_StayDates"].unique()):
            dfpart = pd.DataFrame({"_StayDates":[stay_date]*len(y_pred[i]),
                                   "y_pred":y_pred[i],
                                   "y_true":y_test[i],
                                   "booking_index":[j for j in range(len(y_pred[i]))]})

            dflst.append(dfpart)

        output_df = pd.concat(dflst,axis=0)
        output_df["HotelID"] = hotel_id 
        output_df["pms_sync_off"] = params['WITHOUT_PMS']
        output_df["status"] = "complete"
        output_df["message"] = f"Successfully trained {hotel_id}"
        output_df["eval_start"] = start_date
        output_df["eval_end"] = end_date

        outputs.append(output_df)

# COMMAND ----------

output_all = pd.concat(outputs,axis=0)
timestamp = datetime.datetime.now()
output_all["timestamp"] = str(timestamp)

# COMMAND ----------

output_dataframe = spark.createDataFrame(output_all)

# COMMAND ----------

# Clear table and then run below
spark.sql(f"DROP TABLE IF EXISTS {params['SLIDING_RESULTS_TABLE']}")

# COMMAND ----------

(output_dataframe.write
         .format("delta")
         .mode("append")
         .option("overwriteSchema", "true")
         .saveAsTable(params['SLIDING_RESULTS_TABLE']))

# COMMAND ----------

dbutils.jobs.taskValues.set(key= 'SLIDE_STEP', value = params['SLIDE_STEP'])
dbutils.jobs.taskValues.set(key= 'SLIDING_RESULTS_TABLE', value = params['SLIDING_RESULTS_TABLE'])