import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import train_gpu as train 
import cv2 
import tensorflow as tf



df = pd.read_csv('runs.csv')

raws  = df['Test performance'][0:13]
print(raws)
import ast 

test_accuracies = list()
for i in range(len(raws)):
    x = ast.literal_eval(raws[i])
    test_accuracies.append(x[1][1])
df = df.loc[0:12]
df['test_acc'] = test_accuracies


raws  = df['Test performance'][0:13]
print(raws)
import ast 

test_precision = list()
for i in range(len(raws)):
    x = ast.literal_eval(raws[i])
    test_precision.append(x[2][1])
df['test_prec'] = test_precision

df1 = df[['Name', 'acc', 'val_acc', 'test_acc', 'test_prec', 'Saved Model Name']]
df1.sort_values(['test_acc', 'test_prec'], ascending=False)


jaccards = list()

df1 = df1.sort_values(['Name'], ascending=True)
for run in range(len(df1)):
    row = df1.iloc[run]
    print(row['Name'])

    try:
        tr = train.Trainer(debug_mode=True)
        for feature in ['BS', 'CC', 'GAUS', 'AUG', 'HEQ', 'PCN']:

            if feature in row['Name']:
                print(feature)

                if feature == 'CC':
                    tr.preprocessing_parameters['connected_components'] = True
                if feature == 'GAUS':
                    tr.preprocessing_parameters['gaussian_blur'] = True
                if feature == 'AUG':
                    tr.preprocessing_parameters['augumentation'] = True
                if feature == 'HEQ':
                    tr.preprocessing_parameters['histogram_equalization'] = True
                if feature == 'PCN':
                    tr.preprocessing_parameters['per_channel_normalization'] = True



        tr.load_data()
        tr.build_model()
        tr.model.load_weights(row['Saved Model Name'])
        results = tr.model.predict(tr.data['test_images'])

        jacc_sum = 0
        for res,mask in zip(results, tr.data['test_masks']):
            
            d = mask > 0.5
            r = res
            t, r = cv2.threshold( np.array(r*255, dtype='uint8'), 0, 255, cv2.THRESH_OTSU)
            r = r / 255
            r = r > 0.5

            r = r.flatten()
            d = d.flatten()

            inter = np.logical_and(d,r).sum()
            union = np.logical_or(d,r).sum()



            jacc_sum += inter / union 

        mean_jaccard_index = jacc_sum / len(results)
        print(mean_jaccard_index)
        jaccards.append(mean_jaccard_index)
    
    except:
        jaccards.append(-1)



    

print(jaccards)
df1['jacc'] = jaccards
df1.to_csv('simplified.csv')
print('saved')