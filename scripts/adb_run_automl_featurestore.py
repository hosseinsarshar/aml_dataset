import os
import uuid
import argparse
import pandas as pd
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
from databricks.feature_store import FeatureStoreClient
from databricks.feature_store.client import FeatureStoreClient
from databricks.feature_store.entities.feature_lookup import FeatureLookup
  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split

from azureml.core import Run, Datastore
from azureml.core import Datastore, Dataset

def populate_environ():
    parser = argparse.ArgumentParser(description='Process arguments passed to script')

    # The AZUREML_SCRIPT_DIRECTORY_NAME argument will be filled in if the DatabricksStep
    # was run using a local source_directory and python_script_name
    parser.add_argument('--AZUREML_SCRIPT_DIRECTORY_NAME')

    # Remaining arguments are filled in for all databricks jobs and can be used to build the run context
    parser.add_argument('--AZUREML_RUN_TOKEN')
    parser.add_argument('--AZUREML_RUN_TOKEN_EXPIRY')
    parser.add_argument('--AZUREML_RUN_ID')
    parser.add_argument('--AZUREML_ARM_SUBSCRIPTION')
    parser.add_argument('--AZUREML_ARM_RESOURCEGROUP')
    parser.add_argument('--AZUREML_ARM_WORKSPACE_NAME')
    parser.add_argument('--AZUREML_ARM_PROJECT_NAME')
    parser.add_argument('--AZUREML_SERVICE_ENDPOINT')
    parser.add_argument('--AZUREML_WORKSPACE_ID')
    parser.add_argument('--AZUREML_EXPERIMENT_ID')

    # Arguments that are related to your application
    parser.add_argument("--ground-truth-tbl-name", type=str, dest='ground_truth_tbl_name', default='ground_truth', help="ground-truth-tbl-name")
    
    parser.add_argument("--output_train", type=str, help="output_train directory")
    parser.add_argument("--output_test", type=str, help="output_test directory")
    parser.add_argument("--output_datastore_name", type=str, help="output_datastore_name directory")
    parser.add_argument("--output_train_feature_set_name", type=str, help="output_train_feature_set_name directory")
    parser.add_argument("--output_test_feature_set_name", type=str, help="output_test_feature_set_name directory")

    (args, extra_args) = parser.parse_known_args()
    os.environ['AZUREML_RUN_TOKEN'] = args.AZUREML_RUN_TOKEN
    os.environ['AZUREML_RUN_TOKEN_EXPIRY'] = args.AZUREML_RUN_TOKEN_EXPIRY
    os.environ['AZUREML_RUN_ID'] = args.AZUREML_RUN_ID
    os.environ['AZUREML_ARM_SUBSCRIPTION'] = args.AZUREML_ARM_SUBSCRIPTION
    os.environ['AZUREML_ARM_RESOURCEGROUP'] = args.AZUREML_ARM_RESOURCEGROUP
    os.environ['AZUREML_ARM_WORKSPACE_NAME'] = args.AZUREML_ARM_WORKSPACE_NAME
    os.environ['AZUREML_ARM_PROJECT_NAME'] = args.AZUREML_ARM_PROJECT_NAME
    os.environ['AZUREML_SERVICE_ENDPOINT'] = args.AZUREML_SERVICE_ENDPOINT
    os.environ['AZUREML_WORKSPACE_ID'] = args.AZUREML_WORKSPACE_ID
    os.environ['AZUREML_EXPERIMENT_ID'] = args.AZUREML_EXPERIMENT_ID
    return args, extra_args

def register_output_dataset(ws, output_datastore_name, output, df_feature_data, output_feature_set_name):
  datastore = Datastore(ws, output_datastore_name)

  relative_path_on_datastore = "/azureml/" + output.split('/azureml/')[1] + '/*.parquet'
  print("relative_path_on_datastore")
  print(relative_path_on_datastore)

  dataset = Dataset.Tabular.from_parquet_files(path = [(datastore, relative_path_on_datastore)])

  # Registering Dataset
  preped_data_dtypes = str(df_feature_data.schema.fields)

  now = datetime.now()
  dt_string = now.strftime("%Y-%m-%d %H:%M:%S")

  input_datasets = ["ft_experiments.network",
                    "ft_experiments.customers"]
                    
  tag = {'input_datasets': input_datasets,
          'registered_at': dt_string,
          'run_id': run.parent.id,
          'dtypes': preped_data_dtypes}

  print("tag:")
  print(tag)

  dataset = dataset.register(workspace=ws, 
                                  name=output_feature_set_name, 
                                  description=f'{output_feature_set_name} featurized data',
                                  tags=tag,
                                  create_new_version=True)

  return dataset


def register_at_databricks_featurestore(output_feature_set_name, df_feature):
  fs = FeatureStoreClient()

  print('Registered the feature table')

  feature_table = fs.create_table(
    name=f'ft_experiments.{output_feature_set_name}',
    primary_keys='id',
    schema=df_feature.schema,
    description=f'{output_feature_set_name} featurized data'
  )

  print('Writing the feature table')

  fs.write_table(
    name=f'features.{output_feature_set_name}',
    df = df_feature,
    mode = 'overwrite'
  )

  print('Writing the feature table completed')


if __name__ == "__main__":
  spark = SparkSession.builder.getOrCreate()
  args, extra_args = populate_environ()
  print(f'args: {args}')

  run = Run.get_context(allow_offline=False)
  print(run.parent.id)
  
  fs = FeatureStoreClient()

  spark.sql('use database ft_experiments')

  df = spark.sql(f'select * from {args.ground_truth_tbl_name}')
  print(f'Size of {args.ground_truth_tbl_name} table: [{df.count()}]')

  ## Apply feature lookup on the feature tables:

  feature_lookups = [
        FeatureLookup(
            table_name="ft_experiments.network",
            feature_names=["failure"],
            lookup_key="network"
        ),
        FeatureLookup(
            table_name="ft_experiments.customers",
            feature_names=["salary", "age"],        
            lookup_key="customer")      
    ]

  # create training set:
  training_set = fs.create_training_set(
      df,
      feature_lookups=feature_lookups,
      label="label",
  )
  
  training_df = training_set.load_df()
  print(f'Size of final set: [{training_df.count()}]')

  df_train, df_test = training_df.randomSplit([0.8, 0.2], seed=42)

  print(f'Size of train set: [{df_train.count()}]')
  print(f'Size of test set: [{df_test.count()}]')

  print("output_train", args.output_train)
  print("output_test", args.output_test)
    
  print("output_train_feature_set_name", args.output_train_feature_set_name)
  print("output_test_feature_set_name", args.output_test_feature_set_name)

  ws = run.experiment.workspace

  # Save the dataframe as a Parquet table

  print("Savind df_train and df_test")
  df_train.write.parquet(args.output_train)
  df_test.write.parquet(args.output_test)
  print("df_train and df_test saved")

  dataset_train = register_output_dataset(ws, args.output_datastore_name, args.output_train, df_train, args.output_train_feature_set_name)
  dataset_test = register_output_dataset(ws, args.output_datastore_name, args.output_test, df_test, args.output_test_feature_set_name)
  