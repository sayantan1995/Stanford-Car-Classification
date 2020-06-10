import numpy as np
import pandas as pd 

def clean_file (file):

    car_df = pd.read_csv(file, header= None)
    car_df.columns = ['file','BBOX_x1','BBOX_y1','BBOX_x2','BBOX_y2', 'class']
    names_df = pd.read_csv('names.csv', header= None)
    names_df.columns = ['model_name']
    name_list = []

    for i in range(len(car_df)):
        name = names_df['model_name'][car_df['class'][i] - 1]
        name_list.append(name)

    names = pd.DataFrame(name_list)
    names.columns = ['model_name']
    df = pd.concat([car_df,names], axis= 1) 
    df['BBOX_h'] = (df['BBOX_y2'] - df['BBOX_y1']) + 1
    df['BBOX_w'] = (df['BBOX_x2'] - df['BBOX_x1']) + 1
    df['year'] = df['model_name'].str[-4:]
    df['year'] = df.year.astype('int')
    df['make'] = df['model_name'].str.split(' ').str[0]
    df['type'] = df['model_name'].str.split(' ').str[-2]
    df['model'] = df['model_name'].str.split(' ').str[1]
    df = df[['file','model_name','class','BBOX_x1','BBOX_y1','BBOX_x2','BBOX_y2','BBOX_h','BBOX_w',
                              'year','make','model','type']]
    return df

training = clean_file('anno_train.csv')
training.to_csv('clean_train_data.csv')

testing = clean_file('anno_test.csv')
testing.to_csv('clean_test_data.csv')