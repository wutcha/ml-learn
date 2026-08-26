import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('datasets/heart_disease_risk_dataset_earlymed.csv')

# Chest_Pain,Shortness_of_Breath,Fatigue,Palpitations,Dizziness,Swelling,Pain_Arms_Jaw_Back,Cold_Sweats_Nausea,High_BP,High_Cholesterol,Diabetes,Smoking,Obesity,Sedentary_Lifestyle,Family_History,Chronic_Stress,Gender,Age,Heart_Risk
'''
All yes/no except age and gender
'''

values = df.to_numpy()
risk = values[:,18]
values = values[:,:18]

training_values = values[:60000]
testing_values = values[60000:]

training_risk = risk[:60000]
testing_risk = risk[60000:]

def feature_scaled(value):
    return (value-np.mean(training_values,axis=0))/np.std(training_values,axis=0)

scaled_training_values = feature_scaled(training_values)

weights = np.zeros(len(values[0]))
rate = 0.001
cycles = 2000
bias = 0.0

for _ in range(cycles):
    z = scaled_training_values @ weights + bias
    y = 1/(1 + np.exp(-z))

    #print(np.shape(y),np.shape(risk))

    # binary cross-entropy loss
    loss = -1*np.mean(training_risk * np.log(y) + (1 - training_risk)*np.log(1-y))
    if _%100==0:
        print(f"Cycle: {_} | Error: {loss}")
    '''
    loss = -(risk * log(mx+b) + (1-risk)*log(1-(mx+b)))
    dL/dm = (dL/dy)(dy/dz)(dz/dm)
    dL/dy = -risk/y + (1-risk)/(1-y)
    dy/dz = e^(-z)/(1+e^(-z))^2 = (1-y)*y
    dz/dm = x
    dL/dm = x(-risk(1-y) + (1-risk)y)
    dL/dm = x(y-risk)
    
    '''
    dw = np.mean(scaled_training_values * (y-training_risk).reshape(60000,1), axis = 0)
    db = np.mean(y-training_risk)

    weights -= dw*rate
    bias -= db*rate

    #print(y)

print(weights,bias)

scaled_testing_values = feature_scaled(testing_values)
testing_y = 1/(1+np.exp(-1*(scaled_testing_values @ weights + bias)))
testing_loss = -1*np.mean(testing_risk * np.log(testing_y) + (1 - testing_risk)*np.log(1-testing_y))
testing_y[testing_y>=0.5]=1
testing_y[testing_y<0.5]=0

accuracy = np.sum(np.bitwise_xor(testing_y.astype(int),testing_risk.astype(int)) == 0)/len(testing_risk)
print(f"Testing loss: {testing_loss} | Testing accuracy: {accuracy} {np.sum(testing_risk.astype(int))}")
