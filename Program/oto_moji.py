import speech_recognition as sr
import subprocess
import tempfile
 
import datetime

date = []
textbook = []
orientation = []
activation = []
intension = []
index_ori = []
num = []
from mlask import MLAsk

# 音声入力
while True:
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("何かお話しして下さい。")
        audio = r.listen(source)
    
    try:
        # Google Web Speech APIで音声認識
        text = r.recognize_google(audio, language="ja-JP")
    except sr.UnknownValueError:
        print("Google Web Speech APIは音声を認識できませんでした。")
    except sr.RequestError as e:
        print("GoogleWeb Speech APIに音声認識を要求できませんでした;"
              " {0}".format(e))
    else:
        if text == "終わりだよ":
            break

        print(text)
        emotion_analyzer = MLAsk()
        result = emotion_analyzer.analyze(text)
        if result['emotion'] != None:
            orientation.append(result['orientation'])
            activation.append(result['activation'])
            intension.append(str(result['intension']))
            if result['orientation'] == "POSITIVE":
                index_ori.append(1)
            elif result['orientation'] == "NEGATIVE":
                index_ori.append(-1)
        else:
            orientation.append('')
            activation.append('')
            intension.append('')
            index_ori.append(0)
        dt_now = datetime.datetime.now()
        date.append(str(dt_now))
        textbook.append(text)
        num.append(len(textbook))
        
#csvファイルで保存
import pandas as pd
a = list(range(7))
a[0] = date
a[1] = textbook
a[2] = orientation
a[3] = activation
a[4] = intension
a[5] = index_ori
a[6] = num
df = pd.DataFrame(a)
df.to_csv('oto_moji.csv', encoding='utf_8_sig')

# 認識結果を表示するためのライブラリを読み込む
from matplotlib import pyplot as plt
from PIL import Image
import random
import numpy as np
print("ave_emortion;" + str(sum(num)/len(num)))
#plot
import matplotlib
import matplotlib.pyplot as plt
#plt.plot("x","y","オプション(marker="記号", color = "色", linestyle = "線の種類")")
plt.plot(num, index_ori, color = "blue")
#メモリの設定
plt.xticks(np.arange(1, len(num), step=2))
plt.yticks(np.arange(-1, 1.1, step=1))
#ラベルの設定
plt.xlabel("Count")
plt.ylabel("Emotions")
#補助線の設定
plt.grid(True)
#保存・表示
plt.savefig("oto_result.png")
plt.show()