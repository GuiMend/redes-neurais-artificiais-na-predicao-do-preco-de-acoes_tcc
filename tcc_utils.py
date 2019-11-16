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
    plt.title(
        f'Loss: {history.history["loss"][-1]} / Val_loss: {history.history["val_loss"][-1]}')
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
    df = {'Prediction': predictions_real[:, 0],
          'Real': y_real[:, 0], '% error': error}
    return pd.DataFrame(data=df)


def linear_model(x_train, kernel_initializer, optimizer='adam', loss_function='mean_squared_error'):
    model = Sequential()
    model.add(Dense(1, input_shape=(
        x_train.shape[1],), kernel_initializer=kernel_initializer))
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
    model.add(Dense(layers[0], input_shape=(x_train.shape[1],),
                    activation=activations[0], kernel_initializer=kernel_initializer))
    for i in range(len(layers)-1):
        model.add(Dense(layers[i+1], activation=activations[i+1],
                        kernel_initializer=kernel_initializer))
    model.add(Dense(1, kernel_initializer=kernel_initializer))
    """
    Parameters of the training of the model: objective is to minimize the loss (degree of error, it's what you got wrong)
    A NN is not trying to maximize accuracy, it's always trying to minimize loss, so the way you calculate loss can make
    a huge impact 
    """
    model.compile(optimizer, loss_function)
    return model


def evaluate_model(model, history, x_train, y_train, x_test, y_test, x, y_scaler, model_name='', existing_df=None, val=False, linear=False):
    print(plot_model(model))
    print(model.summary())
    if val:
        plot_loss_with_validation(history)
    else:
        plot_loss(history)
    # Evaluating model against Training and Test set
    train_eval = model.evaluate(x_train, y_train, verbose=0)
    test_eval = model.evaluate(x_test, y_test, verbose=0)

    # Plot linear weights for each column
    if linear:
        print(linear_model_weighs_table(model, x))
    # Predict results in test dataset
    prediction_results = prediction_results_data_frame(
        x_test, y_test, model, y_scaler)

    q1 = prediction_results["% error"].describe()['25%']
    q3 = prediction_results["% error"].describe()['75%']
    inter_quartile_range = q3 - q1
    lower_boundary = q1 - 1.5 * inter_quartile_range
    upper_boundary = q3 + 1.5 * inter_quartile_range

    results_without_outliers = prediction_results[prediction_results['% error'].between(
        lower_boundary, upper_boundary)]
    no_outliers_index = results_without_outliers.index
    test_eval_no_outliers = model.evaluate(
        x_test[no_outliers_index], y_test[no_outliers_index], verbose=0)

    # Print and plot results
    print_and_plot_results(prediction_results)
    plot_model(model)

    print('Remove outliers from results and recalculate loss')

    print_and_plot_results(results_without_outliers)

    print(f'MSE of training: {train_eval}')
    print(f'MSE of testing: {test_eval}')
    print(f'MSE of testing without first outliers: {test_eval_no_outliers}')

    df_result = {'Model Name':  [model_name], 'Testing MSE': test_eval,
                 'Testing MSE after removing outlier': test_eval_no_outliers, 'Training MSE': train_eval}
    df_result = pd.DataFrame(data=df_result)
    if existing_df is None:
        return df_result
    else:
        return existing_df.append(df_result, ignore_index=True)


def print_and_plot_results(prediction_results):
    prediction_results = prediction_results.sort_values(
        by='Real', ascending=False)
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


def loocv(x, y, kfold, kernel_initializer, layers, activations, y_scaler, optimizer, epochs=10, verbose=0, batch_size=5):
    i = 1
    results_loss = np.array([])
    results_evaluate = np.array([])

    df = {'Prediction': [0], 'Real': [0], '% error': [0]}
    results_predictions = pd.DataFrame(data=df)

    for train, test in kfold.split(x):
        print(f'Training K-Fold: {i}/{kfold.get_n_splits()}')
        i += 1

        # Kernel initialization
        kernel_init = kernel_initializer(seed=0)
        # Create model
        model = deep_model(x, kernel_init, layers,
                           activations, optimizer=optimizer)
        # Train model with current fold
        deep_history = model.fit(x[train, :], y[train, :], batch_size=batch_size,
                                 epochs=epochs, verbose=verbose)
        # Save minimum loss encountered for the trained model (MSE)
        results_loss = np.append(
            [results_loss], [[deep_history.history["loss"][-1]]])
        # Evaluate the model with test
        evaluate = model.evaluate(x[test, :], y[test, :], verbose=verbose)
        # Save evaluation loss (MSE)
        results_evaluate = np.append([results_evaluate], [[evaluate]])
        # Predict test y with trained model
        prediction_results = prediction_results_data_frame(
            x[test, :], y[test, :], model, y_scaler)
        # Save results in DataFrame
        results_predictions = results_predictions.append(
            prediction_results, ignore_index=True)

    return results_loss, results_evaluate, results_predictions.iloc[1:, :]


def plot_loss_eval(results_loss, results_evaluate, log=False, font_size=15):
    plt.figure(figsize=(16, 8))
    plt.plot(range(len(results_loss)), results_loss, 'k')
    plt.plot(range(len(results_loss)), results_loss, 'ko')
    plt.plot(range(len(results_evaluate)), results_evaluate, 'r')
    plt.plot(range(len(results_evaluate)), results_evaluate, 'ro')
    plt.legend(['loss', '', 'test_loss', ''], fontsize=font_size)
    plt.xlabel('Fold #', fontsize=font_size)
    plt.ylabel('Loss', fontsize=font_size)
    plt.title(
        f'loss: {results_loss[-1]} / eval_loss: {results_evaluate[-1]}', fontsize=font_size)
    if log:
        plt.yscale('log')
    plt.grid(True)
    plt.show()
