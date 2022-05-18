import os
import uuid
import argparse
import pandas as pd
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
from databricks.feature_store import FeatureStoreClient

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder
from sklearn.compose import make_column_transformer

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
    parser.add_argument("--feature_set_1", type=str, help="input feature set")
    parser.add_argument("--feature_set_2", type=str, help="input feature set")
    parser.add_argument("--feature_set_3", type=str, help="input feature set")
    
    parser.add_argument("--output", type=str, help="output_extract directory")
    parser.add_argument("--output_datastore_name", type=str, help="output_extract directory")
    parser.add_argument("--output_feature_set_name", type=str, help="output_extract directory")

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

def prep_data(data):
    data_train = data.copy()
    gender_labels = {'male':0,'female':1}
    data_train['Sex'] = data_train['Sex'].replace({'male':0,'female':1})

    data_train = data_train.drop(['Name','Ticket','Cabin','Embarked'],axis =1)
    data_train['Age'] = data_train['Age'].fillna(data_train['Age'].mean())
    return data_train

def register_output_dataset(ws, output_datastore_name, output):
  datastore = Datastore(ws, output_datastore_name)

  relative_path_on_datastore = "/azureml/" + output.split('/azureml/')[1]
  print("relative_path_on_datastore")
  print(relative_path_on_datastore)

  delta_file = Dataset.File.from_files(path = [(datastore, relative_path_on_datastore)])

  # Registering Dataset
  preped_data_dtypes = preped_data.dtypes.apply(lambda x: x.name).to_dict()

  now = datetime.now()
  dt_string = now.strftime("%Y-%m-%d %H:%M:%S")

  input_datasets = [f"{ds_feature_set_1.name}: {ds_feature_set_1.version}",
                    f"{ds_feature_set_2.name}: {ds_feature_set_2.version}",
                    f"{ds_feature_set_3.name}: {ds_feature_set_3.version}"]

  tag = {'input_datasets': input_datasets,
          'regisitered_at': dt_string,
          'delta_feature_name': f'features.{args.output_feature_set_name}',
          'run_id': run.parent.id,
          'dtypes': preped_data_dtypes}

  print("tag:")
  print(tag)

  delta_file = delta_file.register(workspace=ws, 
                                  name=args.output_feature_set_name, 
                                  description=f'{args.output_feature_set_name} featurized data',
                                  tags=tag,
                                  create_new_version=True)

  return delta_file

def register_at_databricks_featurestore(output_feature_set_name):
  spark.sql("CREATE SCHEMA IF NOT EXISTS features")

  fs = FeatureStoreClient()

  print('Registered the feature table')

  feature_table = fs.create_table(
    name=f'features.{output_feature_set_name}',
    primary_keys='id',
    schema=df_all.schema,
    description=f'{output_feature_set_name} featurized data'
  )

  print('Writing the feature table')

  fs.write_table(
    name=f'features.{output_feature_set_name}',
    df = df_all,
    mode = 'overwrite'
  )

  print('Writing the feature table completed')

if __name__ == "__main__":
  spark = SparkSession.builder.getOrCreate()
  args, extra_args = populate_environ()

  run = Run.get_context(allow_offline=False)
  print(run.parent.id)

  print("output", args.output)
  print("output_feature_set_name", args.output_feature_set_name)

  ws = run.experiment.workspace

  # Getting the dataset that are passed to the step:

  ds_feature_set_1 = Dataset.get_by_name(ws, name=args.feature_set_1)
  ds_feature_set_2 = Dataset.get_by_name(ws, name=args.feature_set_2)
  ds_feature_set_3 = Dataset.get_by_name(ws, name=args.feature_set_3)

  pdf_feature_set_1 = ds_feature_set_1.to_pandas_dataframe()
  print("pdf_feature_set_1.shape:", pdf_feature_set_1.shape)
  pdf_feature_set_2 = ds_feature_set_2.to_pandas_dataframe()
  print("pdf_feature_set_2.shape:", pdf_feature_set_2.shape)
  pdf_feature_set_3 = ds_feature_set_3.to_pandas_dataframe()
  print("pdf_feature_set_3.shape:", pdf_feature_set_3.shape)

  pdf_all = pd.concat([pdf_feature_set_1,
                      pdf_feature_set_2,
                      pdf_feature_set_3])

  print("pdf_all.shape()")
  print(pdf_all.shape)

  # simple data preprocesing logic to featurize the data

  preped_data = prep_data(pdf_all)

  # adding a unique ID to meet Databricks Feature Store requirement
  preped_data['id'] = preped_data.apply(lambda _: uuid.uuid4().hex, axis=1)

  df_all = spark.createDataFrame(preped_data)

  # Save the dataframe as a Delta table
  print("Savind pdf_all")
  df_all.write \
    .format("delta") \
    .option("overwriteSchema", "true") \
    .mode("overwrite") \
    .save(args.output)
  print("pdf_all saved")

  delta_file = register_output_dataset(ws, args.output_datastore_name, args.output)

  register_at_databricks_featurestore(args.output_feature_set_name)
