from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.optimizers import Adam


def create_model_sensitivity(
    input_shape,
    loss_function="mean_absolute_error",  # Best for volatility regression
    learning_rate=0.001,
    lstm_layers=2,
    activation_function="relu",  # Default now set to 'relu' for intraday use
    output_activation="linear",  # Use 'linear' for regression output
    dropout_rate=0.1,
    units_per_layer=128,  # Default number of units per LSTM layer
):
    '''
    Creates an LSTM model for intraday 15-min volatility forecasting.

    Parameters:
    - input_shape: tuple, e.g., (time_steps, features)
    - loss_function: string, loss to minimize (e.g., 'mean_absolute_error')
    - learning_rate: float, optimizer learning rate
    - lstm_layers: int, number of stacked LSTM layers
    - activation_function: string, LSTM activation ('relu' for fast convergence in intraday)
    - output_activation: string, output Dense activation ('linear' for regression)
    - dropout_rate: float, dropout rate between layers
    - units_per_layer: int, number of LSTM units per layer

    Returns:
    - Compiled Keras Sequential model
    '''
    model = Sequential()

    for i in range(lstm_layers):
        return_seq = i < lstm_layers - 1
        if i == 0:
            model.add(
                LSTM(
                    units_per_layer,
                    activation=activation_function,
                    return_sequences=return_seq,
                    input_shape=input_shape
                )
            )
        else:
            model.add(
                LSTM(
                    units_per_layer,
                    activation=activation_function,
                    return_sequences=return_seq
                )
            )
        model.add(Dropout(dropout_rate))

    model.add(Dense(1, activation=output_activation))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss_function)

    return model