import numpy as np
from sklearn.metrics import mean_squared_error, r2_score 

# Description: Linear Regression using Gradient Descent

class LinearRegGD:
    def __init__(self, Learning_rate=0.001, epochs=1000):
        self.Learning_rate = Learning_rate
        self.epochs = epochs
        self.m = 0
        self.c = 0
    
    def fit(self, x, y):
        ''' 
        x: input data
        y: output data

        '''
        n = float(len(x))
        for i in range(self.epochs):
            y_pred = self.m * x + self.c
            # computing gradients Derivative of m and c
            D_m = (-2/n) * sum(x * (y - y_pred))
            D_c = (-2/n) * sum(y - y_pred)
            # Update parameters
            self.m = self.m - self.Learning_rate * D_m
            self.c = self.c - self.Learning_rate * D_c
        return self
    
    def predict(self, x):
        return self.m * x + self.c
    
# Generate synthetic data
np.random.seed(42)
x = 2 * np.random.rand(100, 1)
y = 4 + 3 * x + np.random.randn(100, 1)

# train test split
x_train, x_test, y_train, y_test = x[:80], x[80:], y[:80], y[80:]

# Model training
model = LinearRegGD()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(y_pred)

# Model Evaluation using Mean Squared Error and r2 square
print("Mean Squared Error: ", mean_squared_error(y_test, y_pred))
print("r2 score: ", r2_score(y_test, y_pred))


