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
from pyspark.sql.types import  TimestampType, StringType, StructField, StructType, DoubleType, LongType, BooleanType, FloatType
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

from mapie.regression import MapieRegressor
from mapie.quantile_regression import MapieQuantileRegressor
from sys import version_info
import mlflow.pyfunc
import cloudpickle
import mlflow
from mlflow.tracking.client import MlflowClient
from autogluon.tabular import TabularDataset, TabularPredictor

import logging
import warnings
from mlflow import MlflowException
import time

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

# COMMAND ----------

print(params)

# COMMAND ----------

warnings.filterwarnings("ignore")
start_time = time.perf_counter()
os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "600"

# COMMAND ----------

cluster_name = spark.conf.get("spark.databricks.clusterUsageTags.clusterName")

if (params['ENV'] == "dev") and ("dev" in cluster_name):
    print(f"Loading phgml package from repo {params['REPOPATH']}")
    sys.path.append(os.path.abspath(params['REPOPATH']))

# COMMAND ----------

from phgml.models.xgboost_model import XGBMultiStepPredictor 
from phgml.models.autogluon_model import AutoGluonModel , AGMlflowModel
from phgml.data.processing_distr import filter_train_data, filter_test_data
from phgml.reporting.output_metrics import *
from phgml.data.data_types import revenue_preprocessed_schema , rooms_preprocessed_schema, training_output_schema

from phgml.reporting.logging import get_logging_path, get_logging_filename
from phgml.reporting.report_results import get_output_df, correct_prediction_list

# COMMAND ----------

# Disable adaptrive query optimization
# Adaptive query optimization groups together smaller tasks into a larger tasks. 
# This may result in limited parallelism if the parallel inference tasks are deemed to be too small by the query optimizer
# We are diableing AQE here to circumevent this limitation on parallelism
spark.conf.set('spark.sql.adaptive.enabled', 'false')

# COMMAND ----------

# Output table schema
training_output_schema = StructType(    
    [
        StructField("HotelID", StringType(), True),
        StructField("pms_sync_off",BooleanType(), True),
        StructField("status", StringType(), True),
        StructField("message", StringType(), True),
        StructField("_StayDates",TimestampType(),True),
        StructField("y_pred",FloatType(),True),
        StructField("y_true",FloatType(),True),
    ]
)

# COMMAND ----------

processing_timestamp = datetime.datetime.now()

logfile_path = get_logging_path(
    params['LOG_ROOT'],
    processing_timestamp)

if not os.path.exists(logfile_path):
    os.makedirs(logfile_path)
    
pms = "PMS"
if params['WITHOUT_PMS']:
    pms = "NOPMS"

log_file_name = get_logging_filename(
    logfile_path,
    "TRAINING",
    params['TARGET_TYPE'],
    pms,
    processing_timestamp)

for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
logging.root.setLevel(logging.INFO)

logger = logging.getLogger(f"training-{params['TARGET_TYPE']}-{pms}")

file_handler = logging.FileHandler(log_file_name)
file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_format)

logger.addHandler(file_handler)
logger.info(f"Processing data for target type: {params['TARGET_TYPE']} : {params['TARGET_COLUMN']}")
logger.info(f"Excluding PMS data? {params['WITHOUT_PMS']}")

# COMMAND ----------

def train_wrapper(
    model_type,
    target_type,
    lag_numbers,
    ml_experiment_id,
    exclude_pms,
    calc_uncertainty,
    prediction_horizon,
    local_root_dir,
    save_matrics=True,
    save_model=True,
    meta_data={},):
    
    def train_data_models(df):
        trainer = None
        hotel_id = df["HotelID"].iloc[0]
        
        test_partition_end = df["_StayDates"].max()
        test_partition_start = test_partition_end -  pd.Timedelta(prediction_horizon,"D")
        meta_data['last_trained_date'] = str(test_partition_start)
        dftrain = filter_train_data(df,test_partition_start)
        dftest = filter_test_data(df,test_partition_start=test_partition_start,test_partition_end=test_partition_end)
        
        model_version = 1
        model_stage = "Staging"
        model_name = None
        pms="PMS"
        if exclude_pms:
            pms = "NOPMS"               
        
        with mlflow.start_run(experiment_id=ml_experiment_id,run_name=f"RUN-{prediction_horizon}days-{hotel_id}-{target_type}-{pms}") as run:
            run_id = run.info.run_id
            if model_type == "XGB":
                trainer = XGBMultiStepPredictor(prediction_horizon=prediction_horizon,
                                                calc_uncertainty=calc_uncertainty,
                                                mlflow_run_id=run_id,
                                                target_type=target_type,
                                                exclude_pms = exclude_pms,
                                                hotel_id=hotel_id)

            else:
                local_root_dir = f"CLF_{prediction_horizon}DAY_AG_{hotel_id}_{target_type}_{pms}/"
                trainer = AutoGluonModel(prediction_horizon=prediction_horizon,
                                         calc_uncertainty=calc_uncertainty,
                                         mlflow_run_id=run_id,
                                         hotel_id=hotel_id,
                                         save_models=save_model,
                                         target_type=target_type,
                                         exclude_pms = exclude_pms,
                                         lag_numbers=lag_numbers,
                                         local_root_dir=local_root_dir,
                                         meta_data = meta_data,) 
                
                trainer.model_type = f"CLF_{prediction_horizon}DAY_AG"
            
            
            output_df = pd.DataFrame()
            try:
                print("training")
                trainer.train(dftrain)

                y_pred , y_test , y_upper , y_lower = trainer.predict(dftest)

                if save_matrics:
                    y_test_flat = [val for ar in y_test for val in ar]
                    y_pred_flat = [val for ar in y_pred for val in ar]
                    
                    SMAPE = mean_absolute_percentage_error(y_test_flat,y_pred_flat,symmetric=True)
                    MAE = mean_absolute_error(y_test_flat,y_pred_flat)

                    mlflow.log_metric(f"SMAPE-{prediction_horizon}", SMAPE)
                    mlflow.log_metric(f"MAE-{prediction_horizon}", MAE)
                    
                dflst = []
                for i,stay_date in enumerate(dftest["_StayDates"].unique()):
                    
                    dfpart = pd.DataFrame({"_StayDates":[stay_date]*len(y_pred[i]),
                               "y_pred":y_pred[i],
                               "y_true":y_test[i]})
        
        
                    dflst.append(dfpart)
                
                
                output_df = pd.concat(dflst,axis=0)
                
                output_df["HotelID"] = hotel_id 
                output_df["pms_sync_off"] = exclude_pms
                output_df["status"] = "complete"
                output_df["message"] = f"Successfully trained {hotel_id}"
                            
                
            except Exception as e:
                raise e
            
                   
        output_df = output_df[["HotelID","pms_sync_off","status","message","_StayDates","y_pred","y_true"]]
        return output_df
    
    return train_data_models

# COMMAND ----------

logger.info(f"Loading data from {params['HIVE_TABLE_NAME']}")
df =  spark.sql(f"select * from {params['HIVE_TABLE_NAME']}").toPandas()

if len(params['SELECTED_HOTELS']) > 0:
    df = df[df["HotelID"].isin(params['SELECTED_HOTELS'])]

df = spark.createDataFrame(df)

# COMMAND ----------

if (df.count() <= 0):
    logger.error("The loaded training dataset is empty.")
    logger.info("Training abort")
    raise Exception("The loaded training dataset is empty.")

# COMMAND ----------

# Serial Training - For testing purposes
"""dft = df.toPandas()

fn = train_wrapper(
        params['MODEL_TYPE'],
        params['TARGET_TYPE'],
        params['LAG_NUMBERS'],
        params['ML_EXPERIMENT_ID'],
        params['WITHOUT_PMS'],
        params['CALC_UNCERTAINTY'],
        params['PREDICTION_HORIZON'],
        params['LOG_ROOT'],
        params['SAVE_METRICS'],
        params['SAVE_MODEL'],
        meta_data = {'training_length': params['PREDICTION_HORIZON'],
                     'inference_length': params['PREDICTION_HORIZON'],})
        #             'tag3': 'value3',})

fn(dft)"""


# COMMAND ----------

# Group the data by hotel id and execute the trainings in parallel
logger.info("Starting parallel training")

output_df = df.groupby("HotelID").applyInPandas(
    train_wrapper(
        params['MODEL_TYPE'],
        params['TARGET_TYPE'],
        params['LAG_NUMBERS'],
        params['ML_EXPERIMENT_ID'],
        params['WITHOUT_PMS'],
        params['CALC_UNCERTAINTY'],
        params['PREDICTION_HORIZON'],
        params['LOG_ROOT'],
        params['SAVE_METRICS'],
        params['SAVE_MODEL'],
        meta_data = {'training_length': params['PREDICTION_HORIZON'],
                     'inference_length': params['PREDICTION_HORIZON'],}),
    training_output_schema)

output_df = output_df.toPandas()

# COMMAND ----------

for index, row in output_df.iterrows():
    if (row.status == "complete"):
        logger.info(f"{row.message}")
    else:
        logger.error(f"Error encountered when training hotel {row.HotelID}: {row.message}")
        
logger.info("Model training completed.")

elapsed_time = time.perf_counter() - start_time
logger.info(f"Time elapsed {elapsed_time}")
logger.info(f"Time elapsed in minutes {elapsed_time/60}")