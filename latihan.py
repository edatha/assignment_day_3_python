import matplotlib.pyplot as plt
import numpy as np

from typing import Tuple


class Model:
    def __init__(self) -> None:
        # initiate random constants a and b
        self.weights = np.random.rand(4)
        self.x_mean = 0
        self.x_std = 1
        self.y_mean = 0
        self.y_std = 1

    def predict(self, x: float) -> float:
        # the input x should normalized first,
        x_norm = (x - self.x_mean) / self.x_std
        # do the prediction
        y_pred_norm = self.calculate_y(x_norm)
        # and the output y should be de-normalized
        return y_pred_norm * self.y_std + self.y_mean
    
    def calculate_y(self, x: float) -> None:
        raise NotImplementedError
    
    def calculate_gradient(self, x: float, y: float, y_pred: float) -> None:
        raise NotImplementedError
    
    def gradient_descent(self, x: float, y: float, learning_rate: float) -> float:
        # First, normalize all input/output so that we get better training
        # normalized_value = (original_value - mean) / standard_deviation
        x = (x - self.x_mean) / self.x_std
        y = (y - self.y_mean) / self.y_std
        # Do the gradient descent
        y_pred = self.calculate_y(x)
        d_weights = self.calculate_gradient(x, y, y_pred)
        self.weights -= learning_rate * d_weights

    def train(self, x_data: float, y_data: float, max_epochs: int = 50, learning_rate: float =0.1) -> Tuple[float, float]:
        print("Start training")

        # First let's calculate the mean and standard deviation for each
        # input x and output y. Those information will be usefil for
        # data normalization.
        self.x_mean = np.mean(x_data)
        self.x_std = np.std(x_data)
        self.y_mean = np.mean(y_data)
        self.y_std = np.std(y_data)
        # Then, perform gradient descent for as much as the number of epochs
        for _ in range(max_epochs):
            self.gradient_descent(x_data, y_data, learning_rate)

        print("Finished training")
    
class LinearModel(Model):
  
    def calculate_y(self, x) -> float:
        # Calculate the y = ax + b
        return self.weights[0] * x + self.weights[1]

    def calculate_gradient(self, x, y, y_pred) -> float:
        # Calculate the gradient
        # dL/da = 2 * x * (pred - actual)
        # dL/db = 2 * (pred - actual)
        d_weights = self.weights.copy()
        d_weights[0] = np.mean(2 * (y_pred - y) * x)
        d_weights[1] = np.mean(2 * (y_pred - y))
        return d_weights

class QuadraticModel(Model):

    def calculate_y(self, x) -> float:
        # Calculate the y = ax^2 + bx + c
        return self.weights[0] * x**2 + self.weights[1] * x + self.weights[2]

    def calculate_gradient(self, x, y, y_pred) -> float:
        # Calculate the gradient
        # dL/da = 2 * x^2 * (pred - actual)
        # dL/db = 2 * x * (pred - actual)
        # dL/dc = 2 * (pred - actual)
        d_weights = self.weights.copy()
        d_weights[0] = np.mean(2 * (y_pred - y) * x**2)
        d_weights[1] = np.mean(2 * (y_pred - y) * x)
        d_weights[2] = np.mean(2 * (y_pred - y))
        return d_weights

class CubicRegressionModel(Model):

    def calculate_y(self, x) -> float:
        # calculate the y = ax^3 + bx^2 + cx + d
        return self.weights[0] * x ** 3 + self.weights[1] * x ** 2 + self.weights[2] * x + self.weights[3]

    def calculate_gradient(self, x, y, y_pred) -> float:
        # calculate the gradient
        # dL/da = 2 * x^3 * (pred - actual)
        # dL/db = 2 * x^2 * (pred - actual)
        # dL/dc = 2 * x * (pred - actual)
        # dL/dd = 2 * (pred - actual)
        d_weights = self.weights.copy()
        d_weights[0] = np.mean(2 * (y_pred - y) * x**3)
        d_weights[1] = np.mean(2 * (y_pred - y) * x**2)
        d_weights[2] = np.mean(2 * (y_pred - y) * x)
        d_weights[3] = np.mean(2 * (y_pred - y))
        return d_weights  
     

def load_data(data_path):
    # load the data as numpy
    data = np.genfromtxt(data_path, delimiter=",")
    # the second column of the data is useless
    data = np.delete(data, 1, 1)
    # there are some NaN rows, remove them
    data = data[~np.isnan(data).any(axis=1)]
    return data


def create_plot(model, x, y, forecast_year):
    # Make model prediction for historical data
    y_pred = model.predict(x)
    # We can calculate the error (MAE) since we have the actual y
    mae = np.mean(np.abs(y - y_pred))

    # The last year that we have in the data
    last_year = x.max()
    # Create year sequences, from the last year in data until target forecast year
    x_forecast = np.arange(last_year, forecast_year + 1)
    y_forecast_pred = model.predict(x_forecast)

    # Write some results text for plot title
    title = "Sea Level Prediction\n"
    title += f"Mean Asolute Error: {mae:.2f} mm\n"
    title += f"Sea Level {int(x_forecast[-1])}: {y_forecast_pred[-1]:.2f} mm"

    # Initialize the plot
    _, ax = plt.subplots(figsize=(8, 8))
    # Draw actual data
    ax.scatter(x, y, label="Actual Data")
    # Draw historical prediction
    ax.plot(x, y_pred, color="red", label="Prediction (Past)")
    # Draw future prediction
    ax.plot(x_forecast, y_forecast_pred, "--", color="red", label="Prediction (Future)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Sea Level (mm)")
    ax.set_title(title)
    ax.legend()
    plt.show()


DATA_PATH = "sea-level.csv"
EPOCHS = 50
LEARNING_RATE = 0.1


def main():
    data = load_data(DATA_PATH)
    x_data, y_data = data[:, 0], data[:, 1]

    forecast_year = int(input("What year do you want to perform forecast? "))
    if forecast_year < x_data[-1]:
        raise ValueError(f"The forecast year must be > {x_data[-1]}")

    model_type = int(
        input("What model degree you want to use (1: linear, 2: quadratic, 3: cubic regression)? ")
    )

    if model_type == 1:
        print("Using linear model")
        model = LinearModel()
        model.train(x_data, y_data, max_epochs=EPOCHS, learning_rate=LEARNING_RATE)
        create_plot(model, x_data, y_data, forecast_year)

    elif model_type == 2:
        print("Using quadratic model")
        model = QuadraticModel()
        model.train(x_data, y_data, max_epochs=EPOCHS, learning_rate=LEARNING_RATE)
        create_plot(model, x_data, y_data, forecast_year)

    elif model_type == 3:
        print("Using cubic regression model")
        model = CubicRegressionModel()
        model.train(x_data, y_data, max_epochs=EPOCHS, learning_rate=LEARNING_RATE)
        create_plot(model, x_data, y_data, forecast_year)

if __name__ == "__main__":
    main()
