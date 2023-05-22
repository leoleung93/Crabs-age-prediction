# -*- coding: utf-8 -*-
"""
AM technical task
prediction_model program
Input:
    The csv file
The CSV file should have the following columns with orders:
Sex, Length, Diameter, Height, Weight, 
Shucked Weight, Viscera Weight, Shell Weight, Age, Predicted Age

Leo Leung
"""

import pandas as pd
import numpy as np
import sys
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def eval_scores(y_true, y_pred):
    '''
    Print the evaluation scores for the model
    Parameters:
        y_true (1d array): true labels array
        y_pred (1d array): predicted values array
    
    Returns:
        0
    '''
    mse = mean_squared_error(y_true,y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true,y_pred)
    
    
    print('Mean Squared Error: {:.5f}'.format(mse))
    print('Root Mean Squard Error: {:.5f}'.format(rmse))
    print('Mean Absolute Error: {:.5f}'.format(mae))
    print('R2 (% of variation explained) : {:.5f}\n'.format(r2))
    return 0




if __name__ == '__main__':
    
    df_path = sys.stdin
    
    #load test data from file
    df = pd.read_csv(df_path)
    text = '''This log file contains the evaluation results of theprediction model for the age of the crab.
    '''
    print(text)
    print('The total numer of testing records: {}\n'.format(len(df)))
    
    y_true = df['Age']
    y_pred = df['Predicted Age']
    eval_scores(y_true,y_pred)
    print('The summary stats of actual data: \n' , y_true.describe())
    print('\nThe summary stats of predicted data: \n' , y_pred.describe())
    