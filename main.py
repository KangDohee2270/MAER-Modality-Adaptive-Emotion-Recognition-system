__author__ = "kdhht5022@gmail.com"
"""
Do make sure: opencv-python: 3.4.1 

using following cmd:
    $pip install opencv-python==3.4.1.0
"""

from multiprocessing import Queue, Value
import configparser, uuid

import asyncio

from torchvision import transforms

from my_util.videomgr import VideoMgr
from my_util.detect_util import draw_results_ssd
from my_util.fer_util import nn_output

import tensorflow as tf

import json

#from spatial_transforms import *
from PIL import Image

import cv2

import pickle

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
import shimmer.New_stcnnmodel as New_stcnnmodel
import shimmer.First_Data_filtering as First_Data_filtering

#hand_detection

from torch.autograd import Variable
from torch.nn import functional as F

from hand.model import generate_model as generate_model
from hand.mean import get_mean, get_std
from hand.opts_inte import parse_opts_offline as parse_opts_offline
from hand.spatial_transforms import *
from hand.target_transforms import ClassLabel as ClassLabel
from my_util import hand_detector_utils as detector_utils

#gaze estimation
from gaze.frame_processor_glasses import frame_processer
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--cam_setting','-c',type=bool,
        default=True, help='Calibrate camera for gaze estimation')
args = parser.parse_args()
print('ok')



gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
  except RuntimeError as e:
    # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
    print(e)


async def async_handle_video(camInfo):
    global VideoHandler
    global config, cam_cap
    global cam_calib, gaze_network, c_data
    global video_set
    config = camInfo["conf"]
    VideoHandler = VideoMgr(int(config['url']), config['name'])
    VideoHandler.open(config)
    cam_cap = VideoHandler.camCtx

    ##For Calibration###
    if not args.cam_setting:
        from gaze.person_calibration import collect_data, fine_tune
        c_data = collect_data(cam_cap, mon, calib_points=9, rand_points=4)

        gaze_network = fine_tune('fer', c_data, frame_processor, mon, device, gaze_network, k, steps=1000, lr=1e-5,
                                 show=False)


    video_set = True
    print("VIDEO SETTING")
    loop = asyncio.get_event_loop()
    global key
    global counting
    global image_queue
    global cam_check

    global do_hand_gesture
    global open_algorithm

    global valence, arousal
    global fd_signal
    global orig_image
    global default_img
    global show_img
    global webcam_size
    h_size = mon.h_pixels
    w_size = mon.w_pixels
    hw_ratio = h_size / w_size
    webcam_size = [int(config['width']), int(config['height'])]
    default_width = webcam_size[0] + 320
    show_img = np.ones([int(default_width * hw_ratio), default_width, 3], dtype=np.uint8)
    default_img = show_img * 255

    face = cv2.imread('icon/happy.png', cv2.IMREAD_COLOR)
    audio = cv2.imread('icon/microphone.png', cv2.IMREAD_COLOR)
    hand = cv2.imread('icon/hold.png', cv2.IMREAD_COLOR)
    body = cv2.imread('icon/running.png', cv2.IMREAD_COLOR)
    bio = cv2.imread('icon/heartbeat.png', cv2.IMREAD_COLOR)

    default_img[70:94, webcam_size[0] + 20: webcam_size[0] + 44] = face
    default_img[180:204, webcam_size[0] + 20: webcam_size[0] + 44] = audio
    default_img[250:274, webcam_size[0] + 20: webcam_size[0] + 44] = bio
    default_img[300:324, webcam_size[0] + 20: webcam_size[0] + 44] = hand
    default_img[340:364, webcam_size[0] + 20: webcam_size[0] + 44] = body

    global input_img

    global emotion_list  # ["angry", "sad", "happy", "pleased", "neutral"]
    global emot_region
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

        global hand_gesture_config
        global hand_gesture_sleep
        global do_face_detect

        global faces
        global net
        global encoder, regressor, disc
        global f

        global valence, arousal
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
            #print('face size : {}'.format(face_size))
            croppted_face_tr = torch.from_numpy(cropped_face.transpose(0, 3, 1, 2)[0] / 255.)  # [3, 224, 224]
            cropped_face_th_norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(croppted_face_tr)

            latent_feature = encoder(cropped_face_th_norm.unsqueeze_(0).type(torch.FloatTensor))  # cuda
            va_output = regressor(latent_feature)

            valence = va_output.detach().cpu().numpy()[0][0] + 0.15
            arousal = va_output.detach().cpu().numpy()[0][1] + 0.15

            fer_a, fer_v = arousal, valence
            do_face_detect = False
        else:
            hand_gesture_sleep = True

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
        # add WangS
        if do_sound:
            chunk = stream.read(CHUNK_SIZE, exception_on_overflow=False)

            aud_raw_data.extend(array('h', chunk))
            aud_index += CHUNK_SIZE
            rec_time = time.time() - record_start

            active = vad.is_speech(chunk, RATE)

            # sys.stdout.write('1' if active else '_')
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
        global aud_raw_data, aud_num, aud_model_v, aud_model_a, aud_predicted_a, aud_predicted_v
        global aud_processing, got_a_sentence
        global audio_valence
        global aud_v, aud_a, got_aud_av
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
            #print(transcript)

            ##predict
            sample = get_feature_vector_from_mfcc(aud_file_name[aud_num], flatten=False)
            sample = sample.reshape(sample.shape[0], sample.shape[1], 1)
            #print(sample.shape)
            aud_v, aud_a = aud_model_v.predict(np.array([sample]))[0], aud_model_a.predict(np.array([sample]))[0]
            got_aud_av = True
            aud_predicted_v = np.argmax(aud_model_v.predict(np.array([sample])))
            aud_predicted_a = np.argmax(aud_model_a.predict(np.array([sample])))

            #print("valence 예측:", str(aud_predicted))


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
        global do_bio, bio_num_list, bio_class_start
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
        global bio_num_list, bio_value, do_bio, bio_model, bio_class_start
        global bio_a, bio_v
        if bio_class_start:
            # preprocessing
            ppg_raw_signal = First_Data_filtering.data_choice(ppg_test_file)  # 처음 10초 데이터 저장
            ppg_signal = First_Data_filtering.ppg_cleaning(ppg_raw_signal)  # 10초 데이터 전처리
            gsr_signal = First_Data_filtering.data_choice(gsr_test_file)
            do_bio=True
            try:
                a = time.time()
                ppg_point = First_Data_filtering.ppg_find_point(ppg_signal)  # 10초 데이터에서 첨두치
                ppg_point_count = ppg_point.__len__()  # PPG 신호 첨두치 개수

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
                return_gsr_signal = First_Data_filtering.gsr_cleaning(y_data_zero)
                a_test, arousal_y_data, v_test, valence_y_data = New_stcnnmodel.CNN_2class(return_ppg_signal, return_gsr_signal,
                                                                       bio_a_model, bio_v_model)
                b = time.time() - a
                print("Bio Processing Time : {}".format(b))
                #print(a_test, v_test)
                bio_a, bio_v = a_test, v_test
                #print(arousal_y_data, valence_y_data)
                bio_value[0] = arousal_y_data[0]
                bio_value[1] = valence_y_data[0]
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

################################################For Gaze Estimation############################################
async def gaze_estimation():
    global arrow_start, arrow_end

    arrow_start = (0, 0)
    arrow_end = (0, 0)

    def gaze_est():
        global arrow_start, arrow_end, gls_flag, do_gaze_est
        if do_gaze_est:
            arrow_start, arrow_end, gls_flag, _ = frame_processor.fer_process(
                'fer', input_img, mon, device, gaze_network, show=True)
            '''try:
            except:
                arrow_start, arrow_end, gls_flag = (0,0), (0,0), 0'''

            do_gaze_est = False

    loop = asyncio.get_event_loop()

    try:
        while (True):
            await loop.run_in_executor(None, gaze_est)
            if key == ord('q'):
                break
    except asyncio.CancelledError:
        pass


###################################################for hand detection##########################################
async def add_hand_detector():
    def hand_detector():
        global detection_graph, sess
        global input_img, img_h, img_w
        global end_face_detection
        global hand_start, hand_end
        global is_hand, hand_size
        global do_hand_detect
        global h_image_batch
        if do_hand_detect:
            image_np = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
            boxes, scores = detector_utils.detect_objects(image_np,
                                                          detection_graph, sess)
            for i in range(num_hands_detect):
                if (scores[i] > opt.score_thresh):
                    (left, right, top, bottom) = (int(boxes[i][1] * img_w),
                                                  int(boxes[i][3] * img_w),
                                                  int(boxes[i][0] * img_h),
                                                  int(boxes[i][2] * img_h))

                    hand_w = right - left
                    if (face_size[0] * face_size[1]) > 55000 or not fd_signal:
                        hand_h = img_h * 0.12
                    else:
                        hand_h = (bottom - top) * 0.2

                    p1 = (max(left - int(hand_w * 0.2), 0), max(top - int(hand_h), 0))
                    p2 = (min(right + int(hand_w * 0.2), img_w), min(bottom + int(hand_h), img_h))

                    hand_start[i] = p1
                    hand_end[i] = p2
                    hand_size[i] = abs((p2[0] - p1[0]) * (p2[1] - p1[1]))
                    # print('{} : {}, {}, {}'.format(i, hand_start[i], hand_end[i], scores[i]))
                    if fd_signal:
                        hf_ratio = (face_size[0] * face_size[1]) / hand_size[i]
                        # print(hf_ratio)
                        if (hf_ratio < 0.70) or ((hand_size[i] / (img_h * img_w)) > 0.6):
                            # print("Hand is too BIG!")
                            is_hand = is_hand + 1
                        else:
                            is_hand = 0
                        if hf_ratio > 5.1:
                            # print("Hand is too SMALL!")
                            hand_start[i] = (0, 0)
                            hand_end[i] = (0, 0)
                        else:
                            crop_hand = input_img[hand_start[i][1]:hand_end[i][1], hand_start[i][0]:hand_end[i][0]]
                            h_image_batch.append(Image.fromarray(crop_hand))
                    else:
                        is_hand = 0
                        crop_hand = input_img[hand_start[i][1]:hand_end[i][1], hand_start[i][0]:hand_end[i][0]]
                        h_image_batch.append(Image.fromarray(crop_hand))
                else:
                    hand_start[i] = (0, 0)
                    hand_end[i] = (0, 0)
                    hand_size[i] = 0

            if hand_size[0] != 0 and hand_size[1] != 0:
                min_start = min(hand_start[0], hand_start[1])
                max_start = max(hand_start[0], hand_start[1])
                min_end = min(hand_end[0], hand_end[1])
                max_end = max(hand_end[0], hand_end[1])

                if (min_end[0] > max_start[0]) and (max_start[1] < min_end[1]):
                    hand_start[1], hand_end[1] = (0, 0), (0, 0)

                    del h_image_batch[-1]
                    '''#print(min_end[0], max_start[0], min_end[1],max_start[1])
                    AoO = (min_end[0] - max_start[0]) * (min_end[1] - max_start[1])
                    AoU = hand_size[0] + hand_size[1] - AoO
                    #print(hand_size[0], hand_size[1], AoO, AoU)
                    if AoO / AoU >  0.2:'''
            do_hand_detect = False

    loop = asyncio.get_event_loop()
    try:
        while (True):
            await loop.run_in_executor(None, hand_detector)
            if key == ord('q'):
                break
    except asyncio.CancelledError:
        pass

###############################################################################################################

##########################################hand geture classification###########################################
async def hand_gesture():
    def gesture_class():
        global spatial_transform, pred, h_image_batch
        global do_hand_class

        if do_hand_class:
            print(len(h_image_batch))
            spatial_transform.randomize_parameters()
            clip = []
            h_image_batch = h_image_batch[:20]
            clip = [spatial_transform(img) for img in h_image_batch]
            print(clip)
            del h_image_batch[:7]
            im_dim = clip[0].size()[-2:]
            do_hand_class = False

            model.eval()

            inputs = torch.cat(clip, 0).view((len(clip), -1) + im_dim).permute(1, 0, 2, 3)
            inputs = inputs.unsqueeze(0)
            #['Point', 'Wave_hand', 'Grasp', 'Thumb_up', 'Thumb_dn']
            with torch.no_grad():
                try:
                    inputs = Variable(inputs)
                    outputs = model(inputs)
                except:
                    return

                '''outputs[0][0] = outputs[0][0] + 3.4
                outputs[0][1] = abs(outputs[0][1]) / 1.4
                outputs[0][2] = abs(outputs[0][2]) / 1.2
                outputs[0][3] = outputs[0][3] / 2.0'''
                #outputs = outputs * 5
                #print(outputs)
                if not opt.no_softmax_in_test:
                    outputs = F.softmax(outputs)
            pred = outputs.argmax(1).cpu().numpy().tolist()

    loop = asyncio.get_event_loop()

    try:
        while (True):
            await loop.run_in_executor(None, gesture_class)

            if key == ord('q'):
                break
    except asyncio.CancelledError:
        pass


###############################################################################################################

#############################################capture & store image#############################################
async def pns_aud_fusion():
    def decision_fusion():
        global got_bio_av, got_aud_av, av_bias, bio_bias, aud_bias
        global guidance_value
        global fus_a, fus_v
        global no_aud_count
        print(no_aud_count)

        if got_aud_av:
            aud_bias = ((np.argmax(aud_v))/2 - 1) * np.max(aud_v)
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
            print('Bio : {}'.format(bio_v))
            if np.max(bio_v) > 0.6:
                print('Bio valence hit : {}'.format(int(np.argmax(bio_v))-1))
                bio_bias = int(np.argmax(bio_v) - 1) * (1 + np.max(bio_v) - 0.6)
            else:
                bio_bias = int(np.argmax(bio_v)) - 1
            got_bio_av = False

        fus_v = (av_bias + bio_bias) / 2
        #print('BIO A : {}'.format(np.argmax(bio_a)))
        if no_aud_count < 5:
            fus_a = ((fer_a + ((np.argmax(aud_a)) / 2 - 1)) + (np.argmax(bio_a)-1)) / 3
        else:
            fus_a = (fer_a + np.argmax(bio_a) - 1 )/ 3


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
        global action
        global something
        global key
        global valence, arousal
        global counting
        global VideoHandler

        global h_image_batch

        ##Flag
        global do_gaze_est
        global do_hand_class
        global do_hand_detect
        global do_face_detect
        global do_body_class
        global do_sound
        global do_bio
        ##
        global emot_region
        global final_emot
        global image_queue
        global cam_check

        global fd_signal

        global orig_image
        global default_img
        global input_img

        global emotion_list  # ["angry", "sad", "happy", "pleased", "neutral"]
        global emot_region
        global pred
        global prev_count
        global show_img

        global stream

        global got_a_sentence, triggered
        global aud_raw_data, start_point, aud_index, record_start
        global ring_buffer, ring_buffer_flags, ring_buffer_flags_end, ring_buffer_index, ring_buffer_index_end

        global bio_num_list, conn
        if open_algorithm and prev_count != counting:
            start_fps = time.time()
            # ----
            # Operate each part
            # ----
            prev_count = counting

            ######Control module#########
            if counting % 1 == 0 :
                if not do_gaze_est:
                    do_gaze_est = True
                if not do_hand_detect :
                    do_hand_detect = True

            if len(h_image_batch) > 20:
                print(len(h_image_batch))
                if not do_hand_class:
                    do_hand_class = True

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

                ring_buffer = collections.deque(maxlen=NUM_PADDING_CHUNKS)
                voiced_frames = []
                ring_buffer_flags = [0] * NUM_WINDOW_CHUNKS
                ring_buffer_index = 0

                ring_buffer_flags_end = [0] * NUM_WINDOW_CHUNKS_END
                ring_buffer_index_end = 0

                # WangS
                aud_raw_data = array('h')
                aud_index = 0
                start_point = 0
                record_start = time.time()
                #print("* 실시간 녹음중 입니다, 현재 버전에서는 종료를 지원하지 않습니다. 강제종료 해주세요.")
                stream.start_stream()

                do_sound = True


            ######Visualizing#########
            if counting % 1 == 0 :
                if type(valence) is torch.Tensor:  # type(valence) is not np.ndarray or
                    valence = valence.detach().cpu().numpy()
                    arousal = arousal.detach().cpu().numpy()
                if np.abs(valence) < 0.1 and np.abs(arousal) < 0.1:
                    final_emot = emotion_list[4]  # neutral
                elif valence > 0.2 and arousal > 0.1:
                    final_emot = emotion_list[2]  # happy
                elif valence < -0.2 and arousal > 0.1:
                    final_emot = emotion_list[0]  # angry
                elif valence < -0.1 and arousal < -0.1:
                    final_emot = emotion_list[1]  # sad
                elif valence > 0.1 and arousal < -0.1:
                    final_emot = emotion_list[3]  # pleased

                if np.sign(valence) == 1 and np.sign(arousal) == 1:
                    emot_region = "1R"
                elif np.sign(valence) == -1 and np.sign(arousal) == 1:
                    emot_region = "2R"
                elif np.sign(valence) == -1 and np.sign(arousal) == -1:
                    emot_region = "3R"
                elif np.sign(valence) == 1 and np.sign(arousal) == -1:
                    emot_region = "4R"

                #                orig_image = await loop.run_in_executor(None, resize, orig_image[:,500:], (400,400))
                #                orig_image_small = await loop.run_in_executor(None, resize, orig_image, (200,200))

                #                if cname[VideoHandler.camName] < 4:
                #                    image_queue.append([cname[VideoHandler.camName], orig_image, orig_image_small])
                #                    cam_check[cname[VideoHandler.camName]] = 1
                #                if len(image_queue) > 50:
                #                    image_queue[:20] = []
                if (config['camshow'] == 'on'):  # & (cname[VideoHandler.camName]%2 == 0 ) & (False):
                    show_img = default_img.copy()
                    face_img = orig_image.copy()

                    for i in range(num_hands_detect):
                        cv2.rectangle(face_img, hand_start[i], hand_end[i], (77, 255, 9), 3, 1)


                    if arrow_start != -1:
                        cv2.arrowedLine(face_img, arrow_start, arrow_end, (255, 0, 0), thickness=1,
                                        line_type=cv2.LINE_AA, tipLength=0.05)

                    if fd_signal == 1:
                        cv2.rectangle(face_img, face_Bbox[0], face_Bbox[1], (255, 77, 9), 3, 1)
                        cv2.putText(show_img, 'Valence: {0:.2f}'.format(np.round(float(valence), 2)), (webcam_size[0]+50, 85),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        cv2.putText(show_img, 'Arousal: {0:.2f}'.format(np.round(float(arousal), 2)), (webcam_size[0]+50, 110),
                                cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)
                        cv2.putText(show_img, '[{}] Emotion: {}'.format(emot_region, final_emot), (webcam_size[0]+50, 145),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                    else:
                        cv2.putText(show_img, 'Face is NOT detected', (webcam_size[0]+50, 85),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                    if aud_processing:
                        cv2.putText(show_img, 'Waiting for Processing', (webcam_size[0] + 50, 197),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    elif (aud_predicted_a != -5) and (aud_predicted_v != -5) and no_aud_count < 5:
                        cv2.putText(show_img, 'Valance : {}'.format(aud_class_list[aud_predicted_v]),
                                    (webcam_size[0] + 50, 182),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        cv2.putText(show_img, 'Arousal : {}'.format(aud_class_list[aud_predicted_a]),
                                    (webcam_size[0] + 50, 212),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    else:
                        cv2.putText(show_img, 'NO AUDIO DATA', (webcam_size[0] + 50, 197),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


                    if bio_value != [-5,-5] :
                        cv2.putText(show_img, 'Valence : {}'.format((bio_value[1]) - 1), (webcam_size[0] + 50, 252),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 50, 200), 2)
                        cv2.putText(show_img, 'Arousal : {}'.format((bio_value[0]) - 1), (webcam_size[0] + 50, 282),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 50, 200), 2)

                    cv2.putText(show_img, 'Valence : {}'.format(round(fus_v,2)), (webcam_size[0] + 50, 352),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 150, 150), 2)
                    cv2.putText(show_img, 'Arousal : {}'.format(round(fus_a,2)), (webcam_size[0] + 50, 382),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 200, 150), 2)

                    if is_hand > 5:
                        cv2.putText(show_img, 'Hand is Too large', (webcam_size[0]+50, 322),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    else :
                        if pred != []:
                            cv2.putText(show_img, '{}'.format(hand_class[pred[0]]), (webcam_size[0] + 50, 322),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)


                    show_img[50:webcam_size[1]+50, 10:webcam_size[0]+10, :] = face_img
            # ----
            # parameter initialization (optional)
            # ---=
            end_fps = 1/(time.time() - start_fps)
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
              + [asyncio.ensure_future(Bio_detect())] \
              + [asyncio.ensure_future(Bio_classification())] \
              + [asyncio.ensure_future(sound())] \
              + [asyncio.ensure_future(Aud_classification())] \
              + [asyncio.ensure_future(gaze_estimation())] \
              + [asyncio.ensure_future(add_hand_detector())] \
              + [asyncio.ensure_future(hand_gesture())] \
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
        global f
        global cam_check

        global cname

        global faces
        faces = np.empty((200, 224, 224, 3))
        global net

        global valence, arousal
        valence, arousal = torch.zeros(1), torch.zeros(1)
        global fd_signal
        fd_signal = 0

        global bio_value
        bio_value = [-5,-5]

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
        global guidance_value
        global fer_a, fer_v
        global fus_a, fus_v
        fer_a, fer_v, fus_a, fus_v = 0.0, 0.0, 0.0, 0.0
        guidance_value = 0.0
        av_bias, bio_bias, aud_bias = 0.0, 0.0, 0.0

        global show_img, img_OK
        img_OK = False
        do_face_detect, do_hand_detect, do_hand_class, do_gaze_est, do_body_class, do_sound, do_bio = False, False, False, False, False, False, False


        global face_size, hand_size
        face_size = []
        hand_size = [0,0]

        # load our serialized face detector from disk
        print("[INFO] loading face detector...")
        protoPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
        modelPath = os.path.sep.join(["face_detector",
                                      "res10_300x300_ssd_iter_140000.caffemodel"])
        net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

        global encoder, regressor, disc

        encoder, regressor, _ = nn_output()
        encoder.load_state_dict(torch.load('weights/AffectNet_KDH_GAN_enc_alexnet_10.t7'), strict=False)
        regressor.load_state_dict(torch.load('weights/AffectNet_KDH_GAN_reg_alexnet_10.t7'), strict=False)

        global emotion_list
        global emot_region
        emotion_list = ["angry", "sad", "happy", "pleased", "neutral"]
        emot_region = ""

        global recording
        global end_aud_class
        recording, end_aud_class = False, False

        encoder.train(False)
        regressor.train(False)
        #    disc.train(False)

        open_algorithm = False
        something = 0;

        key = [0, 0, 0, 0];
        f = 0

        # We can set multiple cameras in the future!
        cname = {'1th_left': 0, }
        #                 '1th_right' : 1,}

        cam_check = [0, 0, 0, 0]

    #        global end_point
    #        global detection_end_point
    #        detection_end_point = step3_modules.init_detection()
    #        end_point = init_net(5)

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
        # self.VideoHdlrProc.stop()
        # self.AnalyzerHdlrProc.stop()
        # self.TrackingHdlrProc.stop()

#####################################################Audio Setting##################################################
def aud_setting():
    global CHUNK_SIZE, NUM_WINDOW_CHUNKS, NUM_WINDOW_CHUNKS_END, RATE, NUM_PADDING_CHUNKS
    global pa, vad
    global ring_buffer, ring_buffer_flags, ring_buffer_flags_end, ring_buffer_index, ring_buffer_index_end
    global aud_raw_data, aud_index, triggered, aud_processing
    global aud_model_v, aud_model_a, aud_start_point
    global aud_file_name, aud_num, aud_predicted_a, aud_predicted_v
    global aud_class_list
    global no_aud_count
    no_aud_count=5
    aud_class_list = [-2, -1, 0, 1, 2]

    global aud_a, aud_v, got_aud_av
    aud_a, aud_v = np.array([0,0,0,0,0]), np.array([0,0,0,0,0])
    got_aud_av = False

    global got_a_sentence
    got_a_sentence = False
    aud_processing = False
    aud_file_name = ['audio/record_0.wav', 'audio/record_1.wav']
    aud_num = 0
    aud_start_point = 0
    aud_predicted_a = -5
    aud_predicted_v = -5

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
    #음성인식 민감도 조절, 1~3
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
    json_file = open(os.path.join('Audio_ER/arousal', 'model.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(os.path.join('Audio_ER/arousal', 'Emotion_Detection_Model.h5'))

    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    aud_model_a = loaded_model

    #load valence model
    json_file = open(os.path.join('Audio_ER/valence', 'model.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(os.path.join('Audio_ER/valence', 'Emotion_Detection_Model.h5'))

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

    global bio_a, bio_v, got_bio_av
    bio_a, bio_v = np.array([0, 1, 0]), np.array([0, 1, 0])
    got_bio_av = False

    bio_csv = 'shimmer_data/{}'.format(bio_name)
    Arousal_model_h5 = "shimmer/Arousal/student_model_20210916.h5"
    Valence_model_h5 = "shimmer/Valence/student_model_20210915.h5"

    bio_a_model = New_stcnnmodel.Arousal_model(Arousal_model_h5)
    bio_v_model = New_stcnnmodel.Valence_model(Valence_model_h5)
    bio_class_start = False

####################################################################################################################
###################################################for gaze_estimation##############################################
# adjust these for your camera to get the best accuracy
# 'v4l2-ctl' is program controls about camera driver
# '-d' option is selecting camera number.
# '-c' option is changing configuration.
def gaze_est_setting():
    global video_set
    global gaze_network, mon, k, frame_processor, device
    #############gaze setting###############
    cam_calib = {'mtx': np.eye(3), 'dist': np.zeros((1, 5))}
    ##NO-calibration Mode
    cam_calib = pickle.load(open("calib_cam0.pkl", "rb"))
    #################################
    # Load gaze network
    #################################
    ted_parameters_path = 'demo_weights/weights_ted.pth.tar'  # weights of DT-ED Network for representation
    maml_parameters_path = 'demo_weights/weights_maml'  # weights of MAML
    k = 9

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create network
    sys.path.append("../gaze/src")
    from models import DTED
    gaze_network = DTED(
        growth_rate=32,
        z_dim_app=64,
        z_dim_gaze=2,
        z_dim_head=16,
        decoder_input_c=32,
        normalize_3d_codes=True,
        normalize_3d_codes_axis=1,
        backprop_gaze_to_encoder=False,
    ).to(device)

    # Load T-ED weights if available
    assert os.path.isfile(ted_parameters_path)
    print('> Loading: %s' % ted_parameters_path)
    ted_weights = torch.load(ted_parameters_path)
    if torch.cuda.device_count() == 1:
        if next(iter(ted_weights.keys())).startswith('module.'):
            ted_weights = dict([(k[7:], v) for k, v in ted_weights.items()])

    #####################################

    # Load MAML MLP weights if available
    full_maml_parameters_path = maml_parameters_path + '/%02d.pth.tar' % k
    assert os.path.isfile(full_maml_parameters_path)
    print('> Loading: %s' % full_maml_parameters_path)
    maml_weights = torch.load(full_maml_parameters_path)
    ted_weights.update({  # rename to fit
        'gaze1.weight': maml_weights['layer01.weights'],
        'gaze1.bias': maml_weights['layer01.bias'],
        'gaze2.weight': maml_weights['layer02.weights'],
        'gaze2.bias': maml_weights['layer02.bias'],
    })
    gaze_network.load_state_dict(ted_weights)

    from gaze.monitor import monitor
    mon = monitor()
    #print(mon)
    frame_processor = frame_processer(cam_calib)
    print("frame process setting complete")
####################################################################################################################

##############################################for hand detector#####################################################
def hand_detector_setting():
    global num_hands_detect
    num_hands_detect = 2
    global detection_graph, sess
    detection_graph, sess = detector_utils.load_inference_graph()
    global end_face_detection  ## for synchronize face & hand detector
    end_face_detection = False

    global hand_start, hand_end  ##손검출 위치 저장을 위한 list
    hand_start = []
    hand_end = []

    for i in range(num_hands_detect):
        hand_start.append((0, 0))
        hand_end.append((0, 0))


####################################################################################################################

###############################################hand gesture recognition#############################################
def hand_gesture_setting():
    global spatial_transform
    global hand_class_start
    global opt
    global model
    global pred
    global hand_class
    hand_class = ['Point', 'Wave_hand', 'Grasp', 'Thumb_up', 'Thumb_dn']
    pred = []
    hand_class_start = False

    opt = parse_opts_offline()
    if opt.root_path != '':
        opt.video_path = os.path.join(opt.root_path, opt.video_path)
        # opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
        opt.result_path = os.path.join(opt.root_path, opt.result_path)
        if opt.resume_path:
            opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
        if opt.pretrain_path:
            opt.pretrain_path = os.path.join(opt.root_path, opt.pretrain_path)
    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    opt.mean = get_mean(opt.norm_value)
    opt.std = get_std(opt.norm_value)

    with open(os.path.join('opts_inte.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    torch.manual_seed(opt.manual_seed)

    model, parameters = generate_model(opt)
    model = model.cuda()

    pytorch_total_params = sum(p.numel() for p in model.parameters() if
                               p.requires_grad)
    print("Total number of trainable parameters: ", pytorch_total_params)

    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)

    spatial_transform = Compose([
        # Scale(opt.sample_size),
        Scale(112),
        CenterCrop(112),
        ToTensor(opt.norm_value), norm_method
    ])
    target_transform = ClassLabel()

    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)
        assert opt.arch == checkpoint['arch']

        opt.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'], strict=False)


####################################################################################################################

if __name__ == "__main__":
    fer_int_alg = FER_INT_ALG()
    print('Start... FER application')
    bio_name = input('Enter the bio filename(must include \'.csv\') : ')
    while not os.path.isfile('shimmer_data/{}'.format(bio_name)):
        bio_name = input('The filename is not valid. Enter the bio filename again(must include \'.csv\') : ')
    bio_setting(bio_name)
    print('BIO setting complete...')
    aud_setting()
    print('Audio setting complete...')
    gaze_est_setting()
    print('gaze estimation setting complete...')
    hand_detector_setting()
    print('hand detector setting complete...')
    hand_gesture_setting()
    print('hand gesture setting complete...')
    fer_int_alg.run()
    fer_int_alg.close()
    print("Completed FER application")
