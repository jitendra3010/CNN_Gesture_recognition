import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics  

def get_acc_loss(history):
    
    loss_train = history.history['loss'][-1]
    accuracy_train = history.history['categorical_accuracy'][-1]

    loss_val = history.history['val_loss'][-1]
    accuracy_val = history.history['val_categorical_accuracy'][-1]

    print(f"\nThe accuracy of the training ::: {accuracy_train:.3f}")
    print(f"The loss of the training ::: {loss_train:.3f}")
    print("\n")
    print(f"The accuracy of the validation ::: {accuracy_val:.3f}")
    print(f"The loss of the validation ::: {loss_val:.3f}")

    return loss_train, accuracy_train, loss_val, accuracy_val

# def getPredClass(y_train, pred_train, y_val, pred_val):
#     # check the prediction classes
#     actual_class_train = y_train.argmax(axis=1)
#     pred_class_train = pred_train.argmax(axis=1)

#     actual_class_val = y_val.argmax(axis=1)
#     pred_class_val = pred_val.argmax(axis=1)

#     return actual_class_train, pred_class_train, actual_class_val, pred_class_val

def getF1Score(actual_class_val, pred_class_val):
    # compute the f1_score
    #f1_train = metrics.f1_score(actual_class_train, pred_class_train, average='weighted')
    f1_val = metrics.f1_score(actual_class_val, pred_class_val, average='weighted')

    #print(f"\nThe F1 Score of the training ::: {f1_train:.3f}")
    print(f"The F1 Score is ::: {f1_val:.3f}")

    return f1_val

def confusionMatrix(actual_class, pred_class, y, modelName):
    '''Function to get the confusion matrix '''
    conf_matrix = confusion_matrix(actual_class, pred_class)
    
    print("\nConfusion Matrix:::")
    print(conf_matrix)

    # Plot confusion matrix as heatmap
    plt.figure(figsize=(6, 3))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=y, yticklabels=y)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(f'Confusion Matrix Heatmap - {modelName}')
    plt.savefig(f'report/confusionMatrix_{modelName}.png')
    plt.show()

    return conf_matrix

def getPredClass(y_actual, y_pred):
    '''Function to get the prediction class'''
    # check the prediction classes
    actual_class = y_actual.argmax(axis=1)
    pred_class = y_pred.argmax(axis=1)

    return actual_class, pred_class

def printClassificationReport(actual_class, pred_class, y):
    'Print the classification Report'
    report = classification_report(actual_class, pred_class, target_names=y)
    print(report)

    return report

def showReport(y_actual, y_pred, modelName):
    '''Funciton to show the report'''

    y = ['LeftSwipe', 'RightSwipe', 'Stop', 'ThumbsDown', 'ThumbsUp']

    #actual_class , pred_class = getPredClass(y_actual, y_pred)
    conf_matrix = confusionMatrix(y_actual, y_pred, y, modelName)
    class_report = printClassificationReport(y_actual, y_pred, y)

    return conf_matrix, class_report