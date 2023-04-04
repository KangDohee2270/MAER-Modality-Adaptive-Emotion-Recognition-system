__author__ = "asrs777@gmail.com"

#######Test for PR########
print("This is example for PR")
from gaze.monitor import monitor
mon = monitor()
from multiprocessing import Queue, Value
import configparser, uuid
import glob
import asyncio

from torchvision import transforms

from my_util.videomgr import VideoMgr
from my_util.detect_util import draw_results_ssd
from my_util.fer_util import nn_output

import tensorflow as tf
import cv2

import torch
import warnings  # control warnings

warnings.filterwarnings("ignore")  # don't print warnings

##########AUDIO MODULE##########

import pyaudio

import webrtcvad
import collections
import sys
from array import array
from struct import pack
from audio.stt import recognize
import wave
import time
import os
import numpy as np
from tensorflow.keras.models import model_from_json
import scipy.io.wavfile as wav
from speechpy.feature import mfcc
###########Bio MODULE##########

import pandas as pd
import bio_signal.stcnnmodel as stcnnmodel
import bio_signal.data_filtering as bio_data_filtering


async def async_handle_video(camInfo):
    global VideoHandler
    global config, cam_cap
    global cam_calib, gaze_network, c_data
    global video_set
    config = camInfo["conf"]
    VideoHandler = VideoMgr(int(config['url']), config['name'])
    VideoHandler.open(config)
    cam_cap = VideoHandler.camCtx
    video_set = True
    print("VIDEO SETTING")
    loop = asyncio.get_event_loop()
    global key
    key = 0
    global counting
    global image_queue
    global cam_check
    global open_algorithm
    global orig_image
    global default_img
    global show_img
    global webcam_size, bezel_size
    global icons
    h_size = mon.h_pixels
    w_size = mon.w_pixels
    hw_ratio = w_size / h_size
    webcam_size = [int(config['width']), int(config['height'])]
    default_height = webcam_size[1]
    show_img = np.zeros([default_height, int(default_height * hw_ratio), 3], dtype=np.uint8)
    default_img = show_img * 255
    bezel_size = int((default_img.shape[1] - webcam_size[0]) / 2)

    face = cv2.imread('icon/happy.png', cv2.IMREAD_COLOR)
    audio = cv2.imread('icon/microphone.png', cv2.IMREAD_COLOR)
    bio = cv2.imread('icon/heartbeat.png', cv2.IMREAD_COLOR)
    fusion = cv2.imread('icon/fusion.png', cv2.IMREAD_COLOR)

    icons = [face, audio, bio, fusion]


    global input_img
    global pred
    global img_w, img_h
    global h_image_batch
    h_image_batch = []
    global is_hand
    is_hand = 0

    global main_start
    main_start = time.time()
    def sleep():
        time.sleep(0.02)
        return

    def resize(img, size):
        return cv2.resize(img, size)

    try:
        counting = 0  # just for counting :)
        show_img = default_img
        while (True):
            if cam_check[cname[VideoHandler.camName]] == 0:
                ret, orig_image = await loop.run_in_executor(None, VideoHandler.camCtx.read)
                orig_image = np.fliplr(orig_image)
                input_img = orig_image
                img_h, img_w, _ = np.shape(input_img)


                counting += 1

                open_algorithm = True

            else:
                await loop.run_in_executor(None, sleep)
                if sum(cam_check) == 4:
                    cam_check = [0, 0, 0, 0]

            if key == ord('q'):
                break

    except asyncio.CancelledError:
        cv2.destroyAllWindows()
        VideoHandler.camCtx.release()


async def add_face_detector():
    def face_detection():
        # global hand_gesture_sleep
        global do_face_detect

        global faces
        global f

        global fd_signal
        global input_img
        global cropped_face
        global face_Bbox, face_size
        global fer_a, fer_v
        if do_face_detect:

            blob = cv2.dnn.blobFromImage(cv2.resize(input_img, (300, 300)), 1.0,
                                         (300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob)
            detected = net.forward()  # For example, (1, 1, 200, 7)

            faces = np.empty((detected.shape[2], 224, 224, 3))
            cropped_face, fd_signal, face_Bbox = draw_results_ssd(detected, input_img, faces, 0.1, 224, img_w, img_h, 0,
                                                                  0, 0)  # 128

            face_size = [face_Bbox[1][0] - face_Bbox[0][0], face_Bbox[1][1] - face_Bbox[0][1]]
            croppted_face_tr = torch.from_numpy(cropped_face.transpose(0, 3, 1, 2)[0] / 255.)  # [3, 224, 224]
            cropped_face_th_norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(croppted_face_tr)

            latent_feature = encoder(cropped_face_th_norm.unsqueeze_(0).type(torch.FloatTensor))  # cuda
            va_output = regressor(latent_feature)

            valence = va_output.detach().cpu().numpy()[0][0] + 0.15
            arousal = va_output.detach().cpu().numpy()[0][1] + 0.15

            fer_a, fer_v = arousal, valence
            do_face_detect = False

    loop = asyncio.get_event_loop()
    try:
        while (True):
            await loop.run_in_executor(None, face_detection)
            if key == ord('q'):
                stream.stop_stream()
                stream.close()
                break
    except asyncio.CancelledError:
        pass

#################################################Audio Recording###############################################
async def sound():
    def record_aud():
        global triggered, got_a_sentence, do_sound, aud_num
        global aud_raw_data, start_point, aud_index, record_start, aud_start_point
        global ring_buffer, ring_buffer_flags, ring_buffer_flags_end, ring_buffer_index, ring_buffer_index_end
        global vad
        if do_sound:
            chunk = stream.read(CHUNK_SIZE, exception_on_overflow=False)

            aud_raw_data.extend(array('h', chunk))
            aud_index += CHUNK_SIZE
            rec_time = time.time() - record_start

            active = vad.is_speech(chunk, RATE)


            ring_buffer_flags[ring_buffer_index] = 1 if active else 0
            ring_buffer_index += 1
            ring_buffer_index %= NUM_WINDOW_CHUNKS

            ring_buffer_flags_end[ring_buffer_index_end] = 1 if active else 0
            ring_buffer_index_end += 1
            ring_buffer_index_end %= NUM_WINDOW_CHUNKS_END

            if not triggered:
                ring_buffer.append(chunk)
                num_voiced = sum(ring_buffer_flags)
                if num_voiced > 0.8 * NUM_WINDOW_CHUNKS:
                    sys.stdout.write(' Open ')
                    triggered = True
                    record_start = time.time()
                    aud_start_point = aud_index - CHUNK_SIZE * 20  # start point
                    # voiced_frames.extend(ring_buffer)
                    ring_buffer.clear()
            # end point detection
            else:
                # voiced_frames.append(chunk)
                ring_buffer.append(chunk)
                num_unvoiced = NUM_WINDOW_CHUNKS_END - sum(ring_buffer_flags_end)
                if num_unvoiced > 0.90 * NUM_WINDOW_CHUNKS_END or rec_time > 10:
                    sys.stdout.write(' Close ')
                    stream.stop_stream()

                    triggered = False
                    got_a_sentence = True
                    do_sound = False
                    stream.close()

            sys.stdout.flush()

    loop = asyncio.get_event_loop()

    try:
        while (True):
            await loop.run_in_executor(None, record_aud)
            if key == ord('q'):
                stream.close()
                break
    except asyncio.CancelledError:
        pass
###############################################################################################################
#################################################Sound Classification##########################################
async def Aud_classification():
    def Aud_class():
        global aud_raw_data, aud_num
        global aud_processing, got_a_sentence
        global audio_valence
        global aud_v, aud_a, aud_prob, got_aud_av
        if got_a_sentence:
            got_a_sentence = False
            aud_processing = True
            ##write to file##
            aud_raw_data.reverse()
            try:
                for index in range(aud_start_point):
                    aud_raw_data.pop()
            except:
                return
            aud_raw_data.reverse()

            MAXIMUM = 32767  # 16384

            ##Normalize
            times = float(MAXIMUM) / max(abs(i) for i in aud_raw_data)
            r = array('h')
            for i in aud_raw_data:
                r.append(int(i * times))
            aud_raw_data = r

            record_to_file(aud_num, aud_raw_data, 2)

            ##predict
            sample = get_feature_vector_from_mfcc(aud_file_name[aud_num], flatten=False)
            sample = sample.reshape(sample.shape[0], sample.shape[1], 1)
            aud_prob = aud_model_v.predict(np.array([sample]))[0]
            aud_v = np.argmax(aud_model_v.predict(np.array([sample]))) - 2
            aud_a = np.argmax(aud_model_a.predict(np.array([sample]))) - 2
            got_aud_av = True

            aud_num = int((aud_num + 1) % 2)
            aud_processing = False
    loop = asyncio.get_event_loop()

    try:
        while (True):
            await loop.run_in_executor(None, Aud_class)
            if key == ord('q'):
                stream.stop_stream()
                stream.close()
                break
    except asyncio.CancelledError:
        pass
###############################################################################################################
#################################################Bio Data Checking#############################################
async def Bio_detect():
    def bio_check():
        global do_bio, bio_class_start
        global ppg_test_file, gsr_test_file
        if do_bio:
            ppg_read = pd.read_csv(bio_csv, usecols=[4], skiprows=[1, 2, 3])
            ppg_test_file = ppg_read.values[:, :]
            ppg_test_file = ppg_test_file.T

            gsr_read = pd.read_csv(bio_csv, usecols=[7], skiprows=[1, 2, 3])
            gsr_test_file = gsr_read.values[:, :]
            gsr_test_file = gsr_test_file.T
            if ppg_test_file[0].__len__() > 10240:  # 10초 데이터만큼 저장되지 않으면 아래의 코드를 실행하지 않음
                bio_class_start = True
                do_bio = False

    loop = asyncio.get_event_loop()

    try:
        while (True):
            await asyncio.sleep(1)
            await loop.run_in_executor(None, bio_check)
            if key == ord('q'):
                stream.stop_stream()
                stream.close()
                break
    except asyncio.CancelledError:
        pass
###############################################################################################################
#############################################Bio Data Classification###########################################
async def Bio_classification():
    def bio_class():
        global do_bio, bio_model, bio_class_start
        global bio_a, bio_v, bio_prob
        if bio_class_start:
            # preprocessing
            ppg_raw_signal = bio_data_filtering.data_choice(ppg_test_file)  # Store initial data(10sec)
            ppg_signal = bio_data_filtering.ppg_cleaning(ppg_raw_signal)  # Preprocessing
            gsr_signal = bio_data_filtering.data_choice(gsr_test_file)
            do_bio=True
            try:
                a = time.time()
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
            except (IndexError, RuntimeWarning):
                print('bio error')

            bio_class_start = False

    loop = asyncio.get_event_loop()

    try:
        while (True):
            await loop.run_in_executor(None, bio_class)
            if key == ord('q'):
                stream.stop_stream()
                stream.close()
                break
    except asyncio.CancelledError:
        pass

#############################################capture & store image#############################################
async def pns_aud_fusion():
    def decision_fusion():
        global got_bio_av, got_aud_av, av_bias, bio_bias, aud_bias
        global guidance_value
        global fus_a, fus_v
        global no_aud_count
        delta, omega = 0.6, 0.5
        print(no_aud_count)

        if got_aud_av:
            aud_bias = (aud_v / 2 - 1) * np.max(aud_prob)
            no_aud_count = 0
            got_aud_av = False
        else:
            no_aud_count += 1

        if no_aud_count < 5:
            av_bias = aud_bias + fer_v
        else:
            av_bias = fer_v
        #print(av_bias)

        if got_bio_av :
            # print('Bio : {}'.format(bio_v))
            bio_hat = np.max(bio_prob) * bio_v

            ex_in_diff = abs(bio_hat - av_bias)
            if ex_in_diff > delta:
                # print('Bio valence hit : {}'.format(int(np.argmax(bio_v))-1))
                bio_bias = (1 + omega * (ex_in_diff - delta)) * bio_hat
            else:
                bio_bias = bio_hat
            got_bio_av = False

        fus_v = (av_bias + bio_bias) / 2
        #print('BIO A : {}'.format(np.argmax(bio_a)))
        if no_aud_count < 5:
            fus_a = (fer_a + aud_bias + bio_a) / 3
        else:
            fus_a = (fer_a + bio_a)/ 3


    loop = asyncio.get_event_loop()
    try:
        while (True):
            if key == ord('q'):
                stream.stop_stream()
                stream.close()
                break
            await loop.run_in_executor(None, decision_fusion)
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass


###############################################################################################################

# agent algorithm
async def Agent():
    global prev_count
    prev_count = -1
    global last_hand_class
    last_hand_class = 0
    global key
    loop = asyncio.get_event_loop()
    def agent() :
        global VideoHandler
        global open_algorithm
        global key
        global counting
        global VideoHandler

        # Basic Flag
        global do_face_detect
        global do_sound
        global do_bio


        global image_queue
        global cam_check

        global orig_image
        global default_img
        global input_img

        global pred
        global prev_count
        global show_img

        global stream

        global got_a_sentence, triggered
        global aud_raw_data, start_point, aud_index, record_start
        global ring_buffer, ring_buffer_flags, ring_buffer_flags_end, ring_buffer_index, ring_buffer_index_end

        global conn

        # for hand
        global h_image_batch


        if open_algorithm and prev_count != counting:
            prev_count = counting

            ######Control module#########
            if counting % 7 == 0:
                if not do_face_detect:
                    do_face_detect = True

            if time.time() - main_start > 10:
                do_bio = True

            if not do_sound:
                stream = pa.open(format=pyaudio.paInt16,
                                 channels=1,
                                 rate=RATE,
                                 input=True,
                                 start=False,
                                 # input_device_index=2,
                                 frames_per_buffer=CHUNK_SIZE)

                triggered = False

                # Start Audio recording

                ring_buffer = collections.deque(maxlen=NUM_PADDING_CHUNKS)
                ring_buffer_flags = [0] * NUM_WINDOW_CHUNKS
                ring_buffer_index = 0

                ring_buffer_flags_end = [0] * NUM_WINDOW_CHUNKS_END
                ring_buffer_index_end = 0

                # WangS
                aud_raw_data = array('h')
                aud_index = 0
                start_point = 0
                record_start = time.time()
                stream.start_stream()

                do_sound = True

            ######Visualizing#########
            if counting % 1 == 0 :
                if (config['camshow'] == 'on'):  # & (cname[VideoHandler.camName]%2 == 0 ) & (False):
                    background = default_img.copy()
                    screen = orig_image.copy()

                    if fd_signal == 1:
                        cv2.rectangle(screen, face_Bbox[0], face_Bbox[1], (255, 77, 9), 3, 1)

                        screen[10:34, 10:34] = icons[0]
                        cv2.putText(screen, '({}, {})'.format(np.round(float(fer_v), 2), np.round(float(fer_a), 2)), (40,30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 102, 255), 2)
                    else:
                        cv2.putText(screen, 'Face is NOT detected', (40,30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 102, 255), 2)

                    screen[40:64, 10:34] = icons[1]
                    if aud_processing:
                        cv2.putText(screen, 'Processing...', (40,60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 153, 255), 2)
                    elif aud_v != -5 and no_aud_count < 5:
                        cv2.putText(screen, '({}, {})'.format(aud_v, aud_a),
                                    (40,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 153, 255), 2)
                    else:
                        cv2.putText(screen, 'NO AUDIO', (40,60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 153, 255), 2)

                    screen[70:94, 10:34] = icons[2]
                    if bio_v != -5 :
                        cv2.putText(screen, '({}, {})'.format(bio_v, bio_a),
                                    (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (51, 153, 0), 2)

                    screen[10:34, (webcam_size[0]-224) : (webcam_size[0]-200)] = icons[3]
                    if bio_v != -5:
                        cv2.putText(screen, '({}, {})'.format(np.round(float(fus_v), 2), np.round(float(fus_a), 2)),
                                    (webcam_size[0]-190, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    print(show_img.shape)
                    background[:, bezel_size:webcam_size[0] + bezel_size, :] = screen
                    show_img = background


            # For checking FPS
            # end_fps = 1/(time.time() - start_fps)
            #print("FINAL FPS = {}".format(end_fps))
    try:
        cv2.namedWindow(config['name'], cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(config['name'], cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        while (True):
            await loop.run_in_executor(None, agent)
            cv2.imshow(config['name'], show_img)
            if VideoHandler.camName == '1th_left':
                key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
                stream.stop_stream()
                stream.close()
                break

    except asyncio.CancelledError:
        pass


async def async_handle_video_run(camInfos):
    futures = [asyncio.ensure_future(async_handle_video(cam_info)) for cam_info in camInfos] \
              + [asyncio.ensure_future(add_face_detector())] \
              + [asyncio.ensure_future(sound())] \
              + [asyncio.ensure_future(Aud_classification())] \
              + [asyncio.ensure_future(Bio_detect())] \
              + [asyncio.ensure_future(Bio_classification())] \
              + [asyncio.ensure_future(pns_aud_fusion())] \
              + [asyncio.ensure_future(Agent())]
    await asyncio.gather(*futures)

class Config():
    """  Configuration for Label Convert Tool """
    def __init__(self):
        global ini
        self.inifile = ini
        self.ini = {}
        self.debug = False
        self.camera_count = 0
        self.cam = []
        self.parser = configparser.ConfigParser()
        self.set_ini_config(self.inifile)

    def set_ini_config(self, inifile):
        self.parser.read(inifile)

        for section in self.parser.sections():
            self.ini[section] = {}
            for option in self.parser.options(section):
                self.ini[section][option] = self.parser.get(section, option)
            if 'CAMERA' in section:
                self.cam.append(self.ini[section])


"""
 *******************************************************************************
 * [CLASS] FER Integration Application
 *******************************************************************************
"""


class FER_INT_ALG():
    """ CVIP Integrated algorithm """

    def __init__(self):
        global ini
        ini = 'config.ini'
        self.Config = Config()
        self.trackingQueue = [Queue() for idx in range(0, int(self.Config.ini['COMMON']['camera_proc_count']))]
        self.vaQueue = Queue()
        self.isReady = Value('i', 0)
        self.camTbl = {}

        global open_algorithm

        global key
        # global f
        global cam_check

        global cname

        global faces
        faces = np.empty((200, 224, 224, 3))
        global net

        global fd_signal
        fd_signal = 0

        global cap_count
        cap_count = 0

        global face_Bbox
        face_Bbox = [(0, 0), (0, 0)]

        global video_set
        video_set = False

        global do_face_detect
        global do_sound
        global do_bio
        global do_gaze_est
        global do_hand_class
        global do_hand_detect

        global av_bias, bio_bias, aud_bias
        global fus_a, fus_v, fer_a, fer_v
        fer_a, fer_v, fus_a, fus_v = 0.0, 0.0, 0.0, 0.0
        av_bias, bio_bias, aud_bias = 0.0, 0.0, 0.0

        global show_img, img_OK
        img_OK = False
        do_face_detect, do_hand_detect, do_hand_class, do_gaze_est, do_body_class, do_sound, do_bio = False, False, False, False, False, False, False


        global face_size, hand_size
        face_size = []
        hand_size = [0,0]

        # load our serialized face detector from disk
        print("[INFO] loading face detector...")
        protoPath = os.path.sep.join(["video/face_detector", "deploy.prototxt"])
        modelPath = os.path.sep.join(["video/face_detector",
                                      "res10_300x300_ssd_iter_140000.caffemodel"])
        net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

        global encoder, regressor

        encoder, regressor, _ = nn_output()
        encoder.load_state_dict(torch.load('video/models/AffectNet_KDH_GAN_enc_alexnet_10.t7', map_location=torch.device('cpu')), strict=False)
        regressor.load_state_dict(torch.load('video/models/AffectNet_KDH_GAN_reg_alexnet_10.t7', map_location=torch.device('cpu')), strict=False)

        global recording
        global end_aud_class
        recording, end_aud_class = False, False

        encoder.train(False)
        regressor.train(False)

        open_algorithm = False
        f = 0

        # We can set multiple cameras in the future!
        cname = {'1th_left': 0, }
        #                 '1th_right' : 1,}

        cam_check = [0, 0, 0, 0]

    def run(self):
        camInfoList = []
        camTbl = {}
        global key
        #        img_data = [[]]*int(self.Config.ini['COMMON']['camera_count'])

        for idx in range(0, int(self.Config.ini['COMMON']['camera_count'])):
            camInfo = {}
            camUUID = uuid.uuid4()

            camInfo.update({"uuid": camUUID})
            camInfo.update({"isready": self.isReady})
            camInfo.update({"tqueue": self.trackingQueue[idx]})
            camInfo.update({"vaqueue": self.vaQueue})
            camInfo.update({"conf": self.Config.cam[idx]})
            camInfo.update({"globconf": self.Config})

            camInfoList.append(camInfo)
            camTbl.update({camUUID: camInfo})

        while (True):
            loop = asyncio.get_event_loop()

            # step1 (mult-camera processing) & step2 (top_k selector)
            # & step3 (top_k frame prediction using SSD)
            loop.run_until_complete(async_handle_video_run(camInfoList))
            loop.close()
            if key == ord('q'):
                break

    def close(self):
        for idx in range(0, int(self.Config.ini['COMMON']['camera_proc_count'])):
            self.trackingQueue[idx].close()
            self.trackingQueue[idx].join_thread()
        self.vaQueue.close()
        self.vaQueue.join_thread()

#####################################################Audio Setting##################################################
def aud_setting():
    global CHUNK_SIZE, NUM_WINDOW_CHUNKS, NUM_WINDOW_CHUNKS_END, RATE, NUM_PADDING_CHUNKS
    global pa, vad
    global ring_buffer, ring_buffer_flags, ring_buffer_flags_end, ring_buffer_index, ring_buffer_index_end
    global aud_raw_data, aud_index, triggered, aud_processing
    global aud_model_v, aud_model_a, aud_start_point
    global aud_file_name, aud_num
    global no_aud_count
    no_aud_count=5

    global aud_a, aud_v, aud_prob
    global got_aud_av
    aud_a, aud_v, aud_prob = -5, -5, []
    got_aud_av = False

    global got_a_sentence
    got_a_sentence = False
    aud_processing = False
    aud_file_name = ['audio/record_0.wav', 'audio/record_1.wav']
    aud_num = 0
    aud_start_point = 0

    RATE = 16000
    CHUNK_DURATION_MS = 10  # supports 10, 20 and 30 (ms)
    PADDING_DURATION_MS = 1500  # 1 sec jugement
    CHUNK_SIZE = int(RATE * CHUNK_DURATION_MS / 1000)  # chunk to read
    NUM_PADDING_CHUNKS = int(PADDING_DURATION_MS / CHUNK_DURATION_MS)
    # NUM_WINDOW_CHUNKS = int(240 / CHUNK_DURATION_MS)
    NUM_WINDOW_CHUNKS = int(400 / CHUNK_DURATION_MS)  # 400 ms/ 30ms  ge
    NUM_WINDOW_CHUNKS_END = NUM_WINDOW_CHUNKS * 2

    try:
        print("기존 파일들을 제거합니다.")
        os.remove(aud_file_name[0])
        os.remove(aud_file_name[1])
    except:
        print("기존파일 제거가 완료되었습니다.")

    vad = webrtcvad.Vad(3)
    pa = pyaudio.PyAudio()



    ring_buffer = collections.deque(maxlen=NUM_PADDING_CHUNKS)
    triggered = False
    voiced_frames = []
    ring_buffer_flags = [0] * NUM_WINDOW_CHUNKS
    ring_buffer_index = 0

    ring_buffer_flags_end = [0] * NUM_WINDOW_CHUNKS_END
    ring_buffer_index_end = 0
    buffer_in = ''
    # WangS
    aud_raw_data = array('h')
    aud_index = 0
    start_point = 0

    ##load pretrained model
    #load arousal model
    json_file = open(os.path.join('audio/models/arousal', 'model.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(os.path.join('audio/models/arousal', 'Emotion_Detection_Model.h5'))

    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    aud_model_a = loaded_model

    #load valence model
    json_file = open(os.path.join('audio/models/valence', 'model.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(os.path.join('audio/models/valence', 'Emotion_Detection_Model.h5'))

    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    aud_model_v = loaded_model


def record_to_file(num, data, sample_width):
    "Records from the microphone and outputs the resulting data to 'path'"
    # sample_width, data = record()
    data = pack('<' + ('h' * len(data)), *data)
    wf = wave.open(aud_file_name[num], 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()

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
####################################################################################################################
###################################################Bio Data Setting#################################################

def bio_setting(bio_name):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

    global bio_class_start
    global bio_a_model, bio_v_model, bio_csv

    global bio_a, bio_v, bio_prob
    global got_bio_av
    bio_a, bio_v = -5, -5
    bio_prob = []
    got_bio_av = False

    bio_csv = bio_name
    Arousal_model_h5 = "bio_signal/models/arousal/student_model_20210916.h5"
    Valence_model_h5 = "bio_signal/models/valence/student_model_20210915.h5"

    bio_a_model = stcnnmodel.Arousal_model(Arousal_model_h5)
    bio_v_model = stcnnmodel.Valence_model(Valence_model_h5)
    bio_class_start = False

####################################################################################################################

if __name__ == "__main__":
    fer_int_alg = FER_INT_ALG()
    print('Start... FER application')
    files = glob.glob('shimmer_data/*.csv')
    files.sort(key=os.path.getmtime, reverse=True)
    bio_name = files[0]
    print(bio_name)
    bio_setting(bio_name)
    print('BIO setting complete...')
    aud_setting()
    print('Audio setting complete...')
    fer_int_alg.run()
    fer_int_alg.close()
    print("Completed FER application")
