import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import plot_model


def plot_loss(history):
    history_df = pd.DataFrame(history.history, index=history.epoch)
    plt.figure(figsize=(8, 6))
    history_df.plot(ylim=(0, history_df.values.max()))
    plt.title(f'Loss: {history.history["loss"][-1]}')
    plt.show()


def plot_loss_with_validation(history):
    history_df = pd.DataFrame(history.history, index=history.epoch)
    plt.figure(figsize=(8, 6))
    history_df.plot(ylim=(0, history_df.values.max()))
    plt.title(f'Loss: {history.history["loss"][-1]} / Val_loss: {history.history["val_loss"][-1]}')
    plt.show()


def linear_model_weighs_table(linear_model, inputs_df):
    linear_wdf = pd.DataFrame(linear_model.get_weights()[0].T,
                              columns=inputs_df.columns).T.sort_values(0, ascending=False)
    linear_wdf.columns = ['feature_weight']
    return linear_wdf


def calculate_diff_percent(a, b):
    return ((100 * (b - a) / b)**2)**.5


def make_array_with_error(x, y):
    array = np.array([])
    for i in range(len(x)):
        array = np.append([array], [[calculate_diff_percent(x[i], y[i])]])
    return array


def prediction_results_data_frame(x, y, model, y_scaler):
    predictions_real = y_scaler.inverse_transform(model.predict(x))
    y_real = y_scaler.inverse_transform(y)
    error = make_array_with_error(predictions_real, y_real)
    df = {'Prediction': predictions_real[:,0], 'Real': y_real[:,0], '% error': error}
    return pd.DataFrame(data=df)


def linear_model(x_train, kernel_initializer, optimizer='adam', loss_function='mean_squared_error'):
    model = Sequential()
    model.add(Dense(1, input_shape=(x_train.shape[1],), kernel_initializer=kernel_initializer))
    model.compile(optimizer, loss_function)
    return model


def deep_model(x_train, kernel_initializer, layers=[32], activations=['relu'], optimizer='adam', loss_function='mean_squared_error'):
    """
    :param x_train: pandas DataFrame - inputs to train the model
    :param kernel_initializer: seeded kernel initializer to make model reproducible
    :param layers: list of integers - output size of the hidden layers - ex: [32, 16, 8]
    :param activations: list of strings - act func of the hidden layers - ex: ['relu', 'relu', 'relu']
    :param optimizer: string - optimizer used to compile model - ex: 'adam'
    :param loss_function: string - ex: 'mean_squared_error'
    :return: compiled model ready to fit

    Regularization helps with generalization of the model
    """
    # Use sequential: it's the basic feed forward NN
    model = Sequential()
    model.add(Dense(layers[0], input_shape=(x_train.shape[1],), activation=activations[0], kernel_initializer=kernel_initializer))
    for i in range(len(layers)-1):
        model.add(Dense(layers[i+1], activation=activations[i+1], kernel_initializer=kernel_initializer))
    model.add(Dense(1, kernel_initializer=kernel_initializer))
    """
    Parameters of the training of the model: objective is to minimize the loss (degree of error, it's what you got wrong)
    A NN is not trying to maximize accuracy, it's always trying to minimize loss, so the way you calculate loss can make
    a huge impact 
    """
    model.compile(optimizer, loss_function)
    return model


def evaluate_model(model, history, x_train, y_train, x_test, y_test, x, y_scaler, val=False, linear=False):
    print(plot_model(model))
    print(model.summary())
    if val:
        plot_loss_with_validation(history)
    else:
        plot_loss(history)
    # Evaluating linear model against Training and Test set
    train_eval = model.evaluate(x_train, y_train, verbose=0)
    test_eval = model.evaluate(x_test, y_test, verbose=0)
    print(f'MSE of training: {train_eval}')
    print(f'MSE of testing: {test_eval}')
    # Plot linear weights for each column
    if linear:
        print(linear_model_weighs_table(model, x))
    # Predict results in test dataset
    prediction_results = prediction_results_data_frame(x_test, y_test, model, y_scaler)
    prediction_results.sort_values(by='Real', ascending=False, inplace=True)
    # Print and plot results
    print(prediction_results)
    plt.plot(prediction_results['Real'], prediction_results['% error'], 'k',
             prediction_results['Real'], prediction_results['% error'], 'ro')
    plt.grid(True)
    plt.title(f'Error - Mean: {prediction_results["% error"].mean()}% / '
              f'Std: {prediction_results["% error"].std()}%')
    plt.xlabel('Stock price')
    plt.ylabel('Error (%)')
    plt.show()
    plt.figure(figsize=(12, 6))
    plt.boxplot(prediction_results["% error"], vert=False)
    plt.title(f'Error - Median: {prediction_results["% error"].median()}%')
    plt.show()
    print(prediction_results["% error"].describe())
    plot_model(model)
