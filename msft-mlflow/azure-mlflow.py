#%% 
import mlflow

#%%
mlflow.set_tracking_uri = "azureml://eastus2.api.azureml.ms/mlflow/v1.0/subscriptions/612cf43d-046d-487c-ab0e-593b3f0ffd50/resourceGroups/jno1387-rg/providers/Microsoft.MachineLearningServices/workspaces/codenamed-josh"
mlflow.set_experiment(experiment_name="heart-condition-classifier")