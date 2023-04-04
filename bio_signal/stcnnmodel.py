import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow import keras
from tensorflow.keras.optimizers import Adam


def Valence_model(model_location_h5):
    Valence_model = load_model(model_location_h5, compile=False, custom_objects={'leaky_relu': tf.nn.leaky_relu})
    return Valence_model


def Arousal_model(model_location_h5):
    Arousal_model = load_model(model_location_h5, compile=False, custom_objects={'leaky_relu': tf.nn.leaky_relu})
    return Arousal_model


def CNN_2class(return_ppg_signal, return_gsr_signal, Arousal_model, Valence_model):
    x_data_ppg = return_ppg_signal.reshape((return_ppg_signal.shape[0], return_ppg_signal.shape[1], 1))
    x_data_gsr = return_gsr_signal.reshape((return_gsr_signal.shape[0], return_gsr_signal.shape[1], 1))

    Arousal_model.compile(optimizer=Adam(), loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          metrics=['accuracy'])
    y_arousal_test = Arousal_model.predict([x_data_ppg, x_data_gsr])
    y_arousal_result = y_arousal_test.argmax(axis=-1)

    Valence_model.compile(optimizer=Adam(), loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          metrics=['accuracy'])
    y_valence_test = Valence_model.predict([x_data_ppg, x_data_gsr])

    y_valence_test[0][0] -= 0.2
    y_valence_test[0][1] += 0.1
    y_valence_test[0][2] += 0.1
    y_valence_result = y_valence_test.argmax(axis=-1)

    return y_arousal_test, y_arousal_result, y_valence_test, y_valence_result
