# -*- coding:utf-8 -*-
import urllib3
import json
import base64
import time


ETRI_URL = "http://aiopen.etri.re.kr:8000/WiseASR/Recognition"
ETRI_Key = "4c306cba-fd2e-4142-a20e-4dc85abca635"
FilePath = 'default/default.wav'
languageCode = "korean"
#languageCode = "english"


def recognize(url=ETRI_URL, key=ETRI_Key, input=FilePath, code=languageCode):
    st = time.time()
    file = open(input, "rb")
    audio_contents = base64.b64encode(file.read()).decode("utf8")
    file.close()

    # audio_contents = base64.b64encode(input).decode("utf8")

    request_json = {
        "access_key": key,
        "argument": {
            "language_code": code,
            "audio": audio_contents
        }
    }

    http = urllib3.PoolManager()
    response = http.request(
        "POST",
        url,
        headers={"Content-Type": "application/json; charset=UTF-8"},
        body=json.dumps(request_json)
    )

    data = response.data
    data = data.decode("utf-8")
    data = data.split('recognized":"')[1]
    data = data.split(' ')
    for i in range(len(data)):
        try:
            data.remove('')
        except ValueError:
            break

    del data[len(data)-1]
    data = " ".join(data)
    et = time.time()
    # print('Confidence: {}'.format(et - st))
    return data