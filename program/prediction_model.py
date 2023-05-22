# -*- coding: utf-8 -*-
"""
AM technical task
prediction_model program
Input:
    The csv file path
The CSV file should have the following columns with orders:
Sex, Length, Diameter, Height, Weight, Shucked Weight, Viscera Weight, Shell Weight, Age

Leo Leung
"""

import pandas as pd
import numpy as np
import pickle
import sys


def data_cleaning(df):
    '''
    Data cleaning and feature engineering
    Parameter:
        df (pd.DataFrame): raw dataset
    
    Return:
        clened df
    '''
    
    #checking the columns name
    cols = ['Sex','Length', 'Diameter', 'Height', 'Weight', 
            'Shucked Weight','Viscera Weight', 'Shell Weight', 'Age']
    assert (df.columns == cols).all(), 'Error: Please check the input columns name'
    
    #cleaning the numerical columns, ensure they are numerical
    num_col = ['Length', 'Diameter', 'Height', 'Weight', 
            'Shucked Weight','Viscera Weight', 'Shell Weight', 'Age']  
    df[num_col] = df[num_col].apply(pd.to_numeric ,errors='coerce')
    
    #remove any missing record
    df.dropna(inplace = True)

    
    #One Hoe Encooding the categorial variable
    Sex_cat = pd.get_dummies(df['Sex'], prefix = 'Sex')
    df = pd.concat([df,Sex_cat],axis = 1)
    df.drop('Sex',axis = 1, inplace = True)
    
    # Feature Engineering
    df['Size'] = df['Length']*df['Diameter']*df['Height']   
    df['Density'] = df['Weight']/df['Size']
    df['Density_shucked'] = df['Shucked Weight']/df['Size']
    df['Density_viscera'] = df['Viscera Weight'] /df['Size']
    df['Weight_no_Shell'] = df['Weight'] - df['Shell Weight']
    df['Density_no_shell'] = df['Weight_no_Shell']/df['Size']
    
    return df


def prediction(df, model_path):
    '''
    Parameters:
        df (pd.DataFrame): raw dataset
        model_path (str): the path of the model(.pkl) to make prediction.
    
    Return:
        1 -d arrary of the prediction values
    ''' 
    #load model from file
 
    model = pickle.load(open(model_path,'rb'))
    y_pred = model.predict(df)
    
    return y_pred

if __name__ == '__main__':
    
    df_path = sys.stdin
 
    sys.stderr.write('Reading model & csv file\n')   
    #df_path = '.\data_clean.csv'
    model_path = './best_xgb.dat'
    
    #load test data from file
    df = pd.read_csv(df_path)
    
    #cleaning the data
    sys.stderr.write('Cleaning dataset\n')
    clean_df = data_cleaning(df)
    
    sys.stderr.write('Make Prediction\n')
    y_pred = prediction(clean_df.drop('Age',axis = 1), model_path)
    
    df['Age'] = df['Age'].astype(int)
    df['Predicted Age'] = np.round(y_pred).astype(int)
    
    df.to_csv(sys.stdout, index = False)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    