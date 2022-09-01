import argparse
import json
import os
import azureml.core
from azureml.core import Workspace, Experiment, Model, Dataset
from azureml.core.model import Model
from azureml.core.resource_configuration import ResourceConfiguration
from azureml.core import Run
from azureml.train.hyperdrive import HyperDriveRun
from shutil import copy2

parser = argparse.ArgumentParser()
parser.add_argument('--saved-model', type=str, dest='saved_model', help='path to saved model file')
parser.add_argument('--model-name', type=str, dest='model_name', help='name of the registered model')
parser.add_argument('--featureset-name-train', type=str, dest='featureset_name_train', default='data', help='training dataset name')
parser.add_argument('--featureset-name-test', type=str, dest='featureset_name_test', default='data', help='test dataset name')

args = parser.parse_args()

model_output_dir = './model/'

os.makedirs(model_output_dir, exist_ok=True)
copy2(args.saved_model, model_output_dir)

run = Run.get_context()
parent_id = run.parent.id
ws = run.experiment.workspace

# ds_feature_train = run.input_datasets['input_train']
# ds_feature_test = run.input_datasets['input_test']
# 
# print('type(ds_feature_train)')
# print(type(ds_feature_train))
# 
# print('type(ds_feature_test)')
# print(type(ds_feature_test))
# 
# print('ds_feature_train')
# print(ds_feature_train)
# 
# print('ds_feature_test')
# print(ds_feature_test)

ds_feature_train = Dataset.get_by_name(ws, name=args.featureset_name_train)
ds_feature_test = Dataset.get_by_name(ws, name=args.featureset_name_test)

model = Model.register(workspace=ws, 
                       datasets=[('featurized training data', ds_feature_train), ('featurized test data', ds_feature_test)], 
                       tags={'run_id': parent_id},
                       description="Trained Model",
                       model_name=args.model_name, 
                       resource_configuration=ResourceConfiguration(cpu=1, memory_in_gb=0.5),
                       model_path=model_output_dir)
