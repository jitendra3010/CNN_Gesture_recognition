import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report  

def confusionMatrix(actual_class, pred_class, y):
    conf_matrix = confusion_matrix(actual_class, pred_class)
    
    print("\nConfusion Matrix:::")
    print(conf_matrix)

    # Plot confusion matrix as heatmap
    plt.figure(figsize=(6, 3))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=y, yticklabels=y)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix Heatmap')
    plt.show()

def getPredClass(y_actual, y_pred):
    # check the prediction classes
    actual_class = y_actual.argmax(axis=1)
    pred_class = y_pred.argmax(axis=1)

    return actual_class, pred_class

def printClassificationReport(actual_class, pred_class, y):
    print(classification_report(actual_class, pred_class, target_names=y))

def showReport(y_actual, y_pred):
    y = ['ThumbsUp', 'ThumnsDown', 'LeftSwipe', 'RightSwipe', 'Stop']

    actual_class , pred_class = getPredClass(y_actual, y_pred)
    confusionMatrix(actual_class, pred_class, y)
    printClassificationReport(actual_class, pred_class, y)