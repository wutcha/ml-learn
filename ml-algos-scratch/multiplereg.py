import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('datasets/Student_Performance.csv')

# Hours Studied, Previous Scores, Extracurricular Activities, Sleep Hours, Sample Question Papers Practiced, Performance Index

class MultipleRegression:
    rate = 0.003
    cycles = 10000
    def __init__(self, rate=0.0005, cycles=10000):
        self.rate = rate
        self.cycles = cycles
        self.weights=None
        self.bias=None
        self.mean=None
        self.std=None

    def fit(self, train_df, target):
        self.mean = np.mean(train_df,axis=0)
        self.std = np.std(train_df,axis=0)

        scaled_values = (train_df-self.mean)/self.std

        self.weights = np.zeros(train_df.shape[1])
        self.bias = 0.0

        for _ in range(self.cycles):
            predictions = (scaled_values @ self.weights + self.bias)
            
            residuals = predictions - target

            loss = np.mean(residuals**2)

            if(_%1000==0):
                print(f"Cycle: {_} | Error: {loss}")

            gradients = np.mean(2 * scaled_values * residuals[:,None], axis=0)
            db = np.mean(2 * residuals)
            
            self.weights -= gradients * self.rate
            self.bias -= db * self.rate

    def predict(self, df):
        scaled_df = (df-self.mean)/self.std
        return scaled_df@self.weights + self.bias

        
    
'''
hours_studied = data['Hours Studied'].to_numpy()
previous_scores = data['Previous Scores'].to_numpy()
ecs = np.where(data['Extracurricular Activities'].to_numpy()=='Yes', 1, 0)
hours_sleep = data['Sleep Hours'].to_numpy()
sample_practice = data['Sample Question Papers Practiced'].to_numpy()

performance = data['Performance Index'].to_numpy()

values = np.array([
    hours_studied,
    previous_scores,
    ecs,
    hours_sleep,
    sample_practice])

'''


# scaled_values = ((training_values-np.mean(training_values,axis=1).reshape(5,1))/(np.std(training_values,axis=1).reshape(5,1)))

#feature_scaled_values = (values-(np.mean(values,axis=1).transpose()))/np.std(values, axis=1).transpose()
# print(scaled_values)
# print(values)

# weights = np.zeros(5)
# bias = 0.0
# rate = 0.0005
# cycles = 10000

# # training

# for _ in range(cycles):
#     predictions = (weights.transpose() @ scaled_values + bias)
#     if(_==0) :
#         print(np.shape(predictions))

#     residuals = predictions - performance[:8000]

#     loss = np.mean(residuals**2)

#     if(_%10==0):
#         print(f"Cycle: {_} | Error: {loss}")

#     gradients = np.mean(2 * scaled_values * residuals, axis=0)
#     db = np.mean(2 * residuals)
#     #print(residuals)
#     #print(values*residuals.transpose())
#     #print(gradients)
    
#     weights -= gradients * rate
#     bias -= db * rate

# testing

# model = MultipleRegression()

# data['Extracurricular Activities'] = data['Extracurricular Activities'].replace({'Yes': 1.0, 'No': 0.0})
# training_answers = data['Performance Index'].to_numpy(dtype=float)
# training_data = data.drop(columns=['Performance Index']).to_numpy(dtype=float)

# model.fit(training_data[:8000],training_answers[:8000])
# y = model.predict(training_data[8000:])
# testing_answers = training_answers[8000:]

# testing_mse = np.mean((testing_answers - y)**2)

# print(f"Weights + bias: {model.weights, model.bias} | Testing MSE: {testing_mse}")








# vibecoded plotting

# fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# feature_names = [
#     "Hours Studied",
#     "Previous Scores",
#     "Extracurricular",
#     "Sleep Hours",
#     "Sample Papers"
# ]

# for i, ax in enumerate(axes.flat[:5]):
#     feature = values[i]
#     fixed = np.mean(values, axis=1)

#     x_plot = np.linspace(feature.min(), feature.max(), 100)
#     plot_values = np.tile(fixed, (100, 1))
#     plot_values[:, i] = x_plot

#     plot_values_scaled = (plot_values - np.mean(values, axis=1)) / np.std(values, axis=1)

#     y_plot = weights @ plot_values_scaled.T + bias

#     ax.scatter(feature, performance, alpha=0.3)
#     ax.plot(x_plot, y_plot, color='red')

#     ax.set_xlabel(feature_names[i])
#     ax.set_ylabel("Performance Index")
#     ax.set_title(f"{feature_names[i]} → Performance")

# axes.flat[5].axis("off")

# plt.tight_layout()
# #plt.show()
