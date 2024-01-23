import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_score, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import mlflow
import os
import shutil
from sklearn.metrics import accuracy_score, precision_score , f1_score, recall_score


def evaluate_and_save_metrics(model, X_test, y_test, save_folder):
    # Calculate ROC AUC score
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC AUC Score: {roc_auc}")

    # Calculate Precision Score
    y_pred = model.predict(X_test)
    precision = precision_score(y_test, y_pred)
    print(f"Precision Score: {precision}")

    # Plot and save ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plot_and_save_roc_curve(fpr, tpr, roc_auc, save_folder)

    # Plot and save Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    average_precision = average_precision_score(y_test, y_pred_proba)
    plot_and_save_precision_recall_curve(precision, recall, average_precision, save_folder)

    # Plot and save Feature Importance Bar Graph
    feature_importances = model.feature_importances_
    feature_names = [f"Feature {i}" for i in range(len(feature_importances))]
    plot_and_save_feature_importance_bar_graph(feature_names, feature_importances, save_folder)


def plot_and_save_roc_curve(fpr, tpr, roc_auc, save_folder):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc='lower right')
    
    save_path = os.path.join(save_folder, 'roc_curve.png')
    plt.savefig(save_path)
    plt.close()

def plot_and_save_precision_recall_curve(precision, recall, average_precision, save_folder):
    plt.figure()
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve: AP={0:0.2f}'.format(average_precision))
    
    save_path = os.path.join(save_folder, 'precision_recall_curve.png')
    plt.savefig(save_path)
    plt.close()

def plot_and_save_feature_importance_bar_graph(feature_names, feature_importances, save_folder):
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, feature_importances, align='center')
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature Name')
    plt.title('Random Forest Feature Importance')
    
    save_path = os.path.join(save_folder, 'feature_importance.png')
    plt.savefig(save_path)
    plt.close()

def save_metrics_to_txt(y_pred, y_test, file_path):
    # Calculate Precision, Recall, and F1 Score
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Save metrics to a text file
    with open(file_path, 'w') as file:
        file.write(f'Precision: {precision}\n')
        file.write(f'Recall: {recall}\n')
        file.write(f'F1 Score: {f1}\n')