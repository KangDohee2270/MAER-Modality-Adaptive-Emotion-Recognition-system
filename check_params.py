from tensorflow.keras.models import model_from_json
import scipy.io.wavfile as wav
from speechpy.feature import mfcc
import json
import os

#####Audio Model
#load arousal model
json_file = open(os.path.join('Audio_ER/arousal', 'model.json'), 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(os.path.join('Audio_ER/arousal', 'Emotion_Detection_Model.h5'))

loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
aud_model_a = loaded_model

print(aud_model_a.summary())

#load valence model
json_file = open(os.path.join('Audio_ER/valence', 'model.json'), 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(os.path.join('Audio_ER/valence', 'Emotion_Detection_Model.h5'))

loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
aud_model_v = loaded_model

print(aud_model_v.summary())

import shimmer.New_stcnnmodel as New_stcnnmodel
Arousal_model_h5 = "shimmer/Arousal/student_model_20210916.h5"
Valence_model_h5 = "shimmer/Valence/student_model_20210915.h5"

bio_a_model = New_stcnnmodel.Arousal_model(Arousal_model_h5)
bio_v_model = New_stcnnmodel.Valence_model(Valence_model_h5)

print(bio_a_model.summary())