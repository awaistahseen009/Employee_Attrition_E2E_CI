'''
We have performed all the experiments using jupyter notebook 
but we cannot use notebook easily in Continous integration so we will
write the code in this training file so its good and easy to train 
during continous integration
'''
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from utils import evaluate_and_save_metrics
from sklearn.metrics import accuracy_score, precision_score , f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from datetime import datetime
from imblearn.over_sampling import SMOTE
current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_name = f'Employee_Attrition_{current_datetime}'
rfc = RandomForestClassifier(n_estimators=50, random_state=2)
mlflow.create_experiment(experiment_name)
mlflow.set_experiment(experiment_name=experiment_name)
mlflow.start_run()
mlflow.sklearn.autolog()
df=pd.read_csv('employee_churn.csv')
df.dropna(inplace=True)
df['salary']=df['salary'].replace({'low':0,'medium':1,'high':2})
X=df.drop(['empid','left'],axis=1)
y=df['left']
x_train,x_test,y_train, y_test=train_test_split(X , y , test_size=.2)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(x_train, y_train)
rfc_resampled=RandomForestClassifier(n_estimators=50, random_state=2)
rfc_resampled.fit(X_resampled, y_resampled)
mlflow.sklearn.save_model(rfc_resampled,f"model_best_v2_resample_{current_datetime}")
mlflow.sklearn.log_model(rfc_resampled, f'Logging_model_v2_resampling')
mlflow.end_run()
last_run = mlflow.last_active_run().info.run_id
eval_data=x_test.copy()
eval_data['left']=y_test
mlflow.evaluate(
f'runs:/{last_run}/Logging_model_v2_resampling',
    data=eval_data,
    targets='left',
    model_type='classifier'
)
destination_folder='metrics_images'
print(x_test.columns)
evaluate_and_save_metrics(rfc_resampled, x_test, y_test, destination_folder)