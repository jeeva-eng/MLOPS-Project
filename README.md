# MLOPS-Project

## Workflows

1.Update config.yaml
2.Update schema.yaml
3.Update params.yaml
4.Update the entity
5.Update the configuration manager  in src config
6.Update the components
7.Update the pipeline
8.Update the main.py
9.Update the app.py


## How to run ?

#### steps:

Clone the repository

'''bash
https://github.com/jeeva-eng/MLOPS-Project
'''

### step 01- Create a conda environment after opening the repository

'''bash
conda create -n mlopsproj python=3.11
'''
'''bash
conda activate mlopsproj
'''

### step 02- install the requirements

'''bash
pip install -r requirements.txt
'''

'''bash
python app.py
'''

Now,
'''bash
open up your local host and port 
'''

## MLflow

[Documentation](https://mlflow.org/docs/latest/index.html)

'''
#### cmd 

-mlflow ui
'''

### dagshub

[dagshub](https://dagshub.com/)


MLFLOW_TRACKING_URI=https://dagshub.com/jeeva-eng/MLOPS-Project.mlflow
MLFLOW_TRACKING_USERNAME=jeeva-eng
MLFLOW_TRACKING_PASSWORD=11f70d0aa3d72d459b18fb5f516f9df6dec7015b
python app.py

Run this to export as env variables:

'''bash

$env:MLFLOW_TRACKING_URI="https://dagshub.com/jeeva-eng/MLOPS-Project.mlflow"
$env:MLFLOW_TRACKING_USERNAME="jeeva-eng"
$env:MLFLOW_TRACKING_PASSWORD=11f70d0aa3d72d459b18fb5f516f9df6dec7015b

'''