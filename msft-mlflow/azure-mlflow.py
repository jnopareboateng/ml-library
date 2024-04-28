#%% 
import mlflow

#%%
mlflow.set_tracking_uri = "azureml://eastus2.api.azureml.ms/mlflow/v1.0/subscriptions/612cf43d-046d-487c-ab0e-593b3f0ffd50/resourceGroups/jno1387-rg/providers/Microsoft.MachineLearningServices/workspaces/codenamed-josh"
mlflow.set_experiment(experiment_name="heart-condition-classifier")
#%%
# example with autologging
from xgboost import XGBClassifier

with mlflow.start_run():
    mlflow.xgboost.autolog()

    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

#%%
# example with custom logging
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

with mlflow.start_run():
    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", accuracy)