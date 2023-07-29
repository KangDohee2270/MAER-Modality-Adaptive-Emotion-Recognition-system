__author__ = "asrs777@gmail.com"

#######Test for PR########
print("This is example for PR")

from torchvision import transforms

from my_util.detect_util import draw_results_ssd
from my_util.fer_util import nn_output

import cv2

import torch
import warnings  # control warnings

warnings.filterwarnings("ignore")  # don't print warnings

##########AUDIO MODULE##########
import os
import numpy as np
import scipy.io.wavfile as wav
from speechpy.feature import mfcc
###########Bio MODULE##########
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import pandas as pd
import bio_signal.stcnnmodel as stcnnmodel
import bio_signal.data_filtering as bio_data_filtering

def get_feature_vector_from_mfcc(file_path: str, flatten: bool,
                                 mfcc_len: int = 39) -> np.ndarray:
    """
    Make feature vector from MFCC for the given wav file.

    Args:
        file_path (str): path to the .wav file that needs to be read.
        flatten (bool) : Boolean indicating whether to flatten mfcc obtained.
        mfcc_len (int): Number of cepestral co efficients to be consider.

    Returns:
        numpy.ndarray: feature vector of the wav file made from mfcc.
    """
    mean_signal_length = 32000


    fs, signal = wav.read(file_path)
    s_len = len(signal)
    # pad the signals to have same size if lesser than required
    # else slice them
    if s_len < mean_signal_length:
        pad_len = mean_signal_length - s_len
        pad_rem = pad_len % 2
        pad_len //= 2
        signal = np.pad(signal, (pad_len, pad_len + pad_rem),
                        'constant', constant_values=0)
    else:
        pad_len = s_len - mean_signal_length
        pad_len //= 2
        signal = signal[pad_len:pad_len + mean_signal_length]
    mel_coefficients = mfcc(signal, fs, num_cepstral=mfcc_len)
    if flatten:
        # Flatten the data
        mel_coefficients = np.ravel(mel_coefficients)
    return mel_coefficients

##Audio model
def audio_model_set():
    ##load pretrained model
    # load arousal model
    global aud_model_a, aud_model_v
    json_file = open(os.path.join('audio/models/arousal', 'model.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(os.path.join('audio/models/arousal', 'Emotion_Detection_Model.h5'))

    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    aud_model_a = loaded_model

    # load valence model
    json_file = open(os.path.join('audio/models/valence', 'model.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(os.path.join('audio/models/valence', 'Emotion_Detection_Model.h5'))

    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    aud_model_v = loaded_model
##Visual model

def visual_model_set():
    print("[INFO] loading face detector...")
    protoPath = os.path.sep.join(["video/face_detector", "deploy.prototxt"])
    modelPath = os.path.sep.join(["video/face_detector",
                                  "res10_300x300_ssd_iter_140000.caffemodel"])

    global encoder, regressor, net
    net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    encoder, regressor, _ = nn_output()
    encoder.load_state_dict(
        torch.load('video/models/AffectNet_KDH_GAN_enc_alexnet_10.t7', map_location=torch.device('cpu')), strict=False)
    regressor.load_state_dict(
        torch.load('video/models/AffectNet_KDH_GAN_reg_alexnet_10.t7', map_location=torch.device('cpu')), strict=False)

    global recording
    global end_aud_class
    recording, end_aud_class = False, False

    encoder.train(False)
    regressor.train(False)

    open_algorithm = False
##Bio model
def bio_model_set():
    Arousal_model_h5 = "bio_signal/models/arousal/student_model_20210916.h5"
    Valence_model_h5 = "bio_signal/models/valence/student_model_20210915.h5"
    global bio_a_model, bio_v_model
    bio_a_model = stcnnmodel.Arousal_model(Arousal_model_h5)
    bio_v_model = stcnnmodel.Valence_model(Valence_model_h5)


def get_face_signal(input_vid):
    cap = cv2.VideoCapture(input_vid)
    valences, arousals = [], []
    while True:
        ret, input_img = cap.read()
        if ret:

            # print(input_img.shape)
            img_h, img_w, _ = input_img.shape
            blob = cv2.dnn.blobFromImage(cv2.resize(input_img, (300, 300)), 1.0,
                                         (300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob)
            detected = net.forward()  # For example, (1, 1, 200, 7)

            faces = np.empty((detected.shape[2], 224, 224, 3))
            cropped_face, fd_signal, face_Bbox = draw_results_ssd(detected, input_img, faces, 0.1, 224, img_w, img_h, 0,
                                                                  0, 0)  # 128
            croppted_face_tr = torch.from_numpy(cropped_face.transpose(0, 3, 1, 2)[0] / 255.)  # [3, 224, 224]
            cropped_face_th_norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(croppted_face_tr)

            latent_feature = encoder(cropped_face_th_norm.unsqueeze_(0).type(torch.FloatTensor))  # cuda
            va_output = regressor(latent_feature)

            valence = va_output.detach().cpu().numpy()[0][0] + 0.15
            arousal = va_output.detach().cpu().numpy()[0][1] + 0.15

            valences.append(valence)
            arousals.append(arousal)
        else:
            break

    fer_a, fer_v = sum(arousals) / len(arousals), sum(valences) / len(valences)
    return fer_a, fer_v

def get_audio_signal(aud_file_wav):
    sample = get_feature_vector_from_mfcc(aud_file_wav, flatten=False)
    sample = sample.reshape(sample.shape[0], sample.shape[1], 1)
    aud_prob = aud_model_v.predict(np.array([sample]))[0]
    aud_v = np.argmax(aud_model_v.predict(np.array([sample]))) - 2
    aud_a = np.argmax(aud_model_a.predict(np.array([sample]))) - 2

    return aud_prob, aud_v, aud_a

def get_bio_signal(bio_csv):
    # Get ppg file
    ppg_read = pd.read_csv(bio_csv, usecols=[4], skiprows=[1, 2, 3])
    ppg_test_file = ppg_read.values[:, :]
    ppg_test_file = ppg_test_file.T
    # Get gsr file
    gsr_read = pd.read_csv(bio_csv, usecols=[7], skiprows=[1, 2, 3])
    gsr_test_file = gsr_read.values[:, :]
    gsr_test_file = gsr_test_file.T

    # Get valence, arousal score
    ppg_raw_signal = bio_data_filtering.data_choice(ppg_test_file)  # Store initial data(10sec)
    ppg_signal = bio_data_filtering.ppg_cleaning(ppg_raw_signal)  # Preprocessing
    gsr_signal = bio_data_filtering.data_choice(gsr_test_file)

    ppg_point = bio_data_filtering.ppg_find_point(ppg_signal)  # Get peak singals
    ppg_point_count = ppg_point.__len__()  # number of peak signals

    half_window = int((10240 / ppg_point_count) / 2)
    if half_window > 550:
        half_window = 550

    if ppg_point[ppg_point_count - 1] < (10240 - half_window):  # 마지막 peak 이후 550보다 많이 남으면
        last_ppg_point = ppg_point[ppg_point_count - 1]
    else:
        last_ppg_point = ppg_point[ppg_point_count - 2]

    x_data_zero = np.zeros((1, 1100))
    y_data_zero = np.zeros((1, 1100))

    for k in range(2 * half_window):
        x_data_zero[0][550 - half_window + k] = ppg_signal[0][last_ppg_point - half_window + k]
        y_data_zero[0][550 - half_window + k] = gsr_signal[0][last_ppg_point - half_window + k]
    return_ppg_signal = x_data_zero  # 최종
    return_gsr_signal = bio_data_filtering.gsr_cleaning(y_data_zero)
    a_test, arousal_y_data, v_test, valence_y_data = stcnnmodel.CNN_2class(return_ppg_signal, return_gsr_signal,
                                                                           bio_a_model, bio_v_model)

    bio_prob = v_test
    bio_a, bio_v = np.argmax(a_test) - 1, np.argmax(v_test) - 1

    return bio_prob, bio_a, bio_v

def adaptive_fusion(fer_a, fer_v, aud_prob, aud_a, aud_v, bio_prob, bio_a, bio_v):
    delta, omega = 0.6, 0.5

    aud_bias = (aud_v / 2 - 1) * np.max(aud_prob)
    av_bias = aud_bias + fer_v

    bio_hat = np.max(bio_prob) * bio_v

    ex_in_diff = abs(bio_hat - av_bias)
    if ex_in_diff > delta:
        # print('Bio valence hit : {}'.format(int(np.argmax(bio_v))-1))
        bio_bias = (1 + omega * (ex_in_diff - delta)) * bio_hat
    else:
        bio_bias = bio_hat

    fus_v = (av_bias + bio_bias) / 2

    fus_a = (fer_a + aud_a + bio_a) / 3

    return fus_v, fus_a


if __name__ == "__main__":
    audio_model_set()
    bio_model_set()
    visual_model_set()

    video_file, bio_file, audio_file = "example_record/example_vid.mp4", "example_record/example_bio.csv", "example_record/example_aud.wav"
    fer_a, fer_v = get_face_signal(video_file)
    print("Complete FER")
    ber_prob, ber_a, ber_v = get_bio_signal(bio_file)
    print("Complete BER")
    ser_prob, ser_v, ser_a = get_audio_signal(audio_file)
    print("Complete SER")
    mer_v, mer_a = adaptive_fusion(fer_a, fer_v, ser_prob, ser_a, ser_v, ber_prob, ber_a, ber_v)

    ser_cont, ber_cont = (ser_v / 2 - 1) * np.max(ser_prob), np.max(ber_prob) * ber_v
    print(f"FER: {round(fer_v, 3)} / SER: {round(ser_cont, 3)}/ BER: {round(ber_cont, 3)} / AF_result: {mer_v}")


    with open("results/example_result.txt", "w") as file:
        file.write("Example Result\n")
        file.write(f"Used File: {video_file} for visual input, {audio_file} for audio input, and {bio_file} for bio input\n")
        file.write(f"Result - FER: {round(fer_v, 3)} / SER: {round(ser_cont, 3)}/ BER: {round(ber_cont, 3)} / AF_result: {round(mer_v, 3)}")
