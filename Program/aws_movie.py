import cv2
import numpy as np
import boto3

# Define ENDPOINT, CLIENT_ID, PATH_TO_CERT, PATH_TO_KEY, PATH_TO_ROOT, MESSAGE, TOPIC, and RANGE
ENDPOINT = "a11dq6o28la5x0-ats.iot.ap-northeast-1.amazonaws.com"
CLIENT_ID = "testDevice"
PATH_TO_CERT = "setting/8f6c903c31555313ea9c21f6cd301ee4200005b2035c833390bf84a757527214-certificate.pem.crt"
PATH_TO_KEY = "setting/8f6c903c31555313ea9c21f6cd301ee4200005b2035c833390bf84a757527214-private.pem.key"
PATH_TO_ROOT = "setting/AmazonRootCA1.pem"
MESSAGE = "Hello World"
TOPIC = "data/test"

# スケールや色などの設定
scale_factor = .15
green = (0,255,0)
red = (0,0,255)
frame_thickness = 2
cap = cv2.VideoCapture(0)
rekognition = boto3.client('rekognition')

# フォントサイズ
fontscale = 1.0
# フォント色 (B, G, R)
color = (0, 120, 238)
# フォント
fontface = cv2.FONT_HERSHEY_DUPLEX

import datetime

# Rekognitionの認識結果を配列で保存
#人と犬の位置情報を格納する配列の作成
Dtime = []
per1 = [] 
per11 = []
check1 = []
per2 = [] 
per21 = []
check2 = []
per3 = [] 
per31 = []
check3 = []
per4 = [] 
per41 = []
check4 = []
per5 = []
per51 = []
check5 = []
per61 = []
per6 = []
check6 = []
per71 = []
per7 = []
check7 = []
per81 = []
per8 = []
check8 = []
per_result = []
ti = []
#n = 1
num = 0
hikensya = input("被験者：")
junban = input("回数：")
import time

start_time = time.time()
end = time.time()
while(end - start_time < 5):
    time.sleep(1)
    print(str((5- (end -start_time))))
    end = time.time()
end = time.time()
print("start")
#fpsを20.0にして撮影したい場合はfps=20.0にします
fps = 5.0
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))#カメラの幅を取得
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))#カメラの高さを取得
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')#動画保存時の形式を設定
name = hikensya + "_video_" + junban + ".mp4"#保存名　video + 現在時刻 + .mp4
video = cv2.VideoWriter(name, fourcc, fps, (w,h))#(保存名前、fourcc,fps,サイズ)

while(5 <= (end - start_time) < 11):
    # フレームをキャプチャ取得
    ret, frame = cap.read()
    video.write(frame)#1フレーム保存する
    height, width, channels = frame.shape
    
    # jpgに変換 画像ファイルをインターネットを介してAPIで送信するのでサイズを小さくしておく
    small = cv2.resize(frame, (int(width * scale_factor), int(height * scale_factor)))
    ret, buf = cv2.imencode('.jpg', small)
    
    # Amazon RekognitionにAPIを投げる
    faces = rekognition.detect_faces(Image={'Bytes':buf.tobytes()}, Attributes=['ALL'])
    
    # 顔の周りに箱を描画する
    for face in faces['FaceDetails']:
        num = num + 1
        dt_now = datetime.datetime.now()
        smile = face['Smile']['Value']
        cv2.rectangle(frame,
                    (int(face['BoundingBox']['Left']*width),
                    int(face['BoundingBox']['Top']*height)),
                    (int((face['BoundingBox']['Left']+face['BoundingBox']['Width'])*width),
                    int((face['BoundingBox']['Top']+face['BoundingBox']['Height'])*height)),
                    green if smile else red, frame_thickness)
        emothions = face['Emotions']
        i = 0
        for emothion in emothions:
            cv2.putText(frame,
                        str(emothion['Type']) + ": " + str(emothion['Confidence']),                           
                        (25, 40 + (i * 25)),
                        fontface,
                        fontscale,
                        color)
            i += 1
            if i == 1:
                n = 1
            elif i == 2:
                n = 0
            if str(emothion['Type']) == 'HAPPY':
                Dtime.append(str(dt_now))
                ti.append(num)
                per1.append(str(emothion['Confidence']))
                per11.append(emothion['Confidence'])
                check1.append(emothion['Confidence'])
            elif str(emothion['Type']) == 'CONFUSED':
                per2.append(str(emothion['Confidence']))
                per21.append(emothion['Confidence'])
                check2.append(emothion['Confidence'])
            elif str(emothion['Type']) == 'SURPRISED':
                per3.append(str(emothion['Confidence']))
                per31.append(emothion['Confidence'])
                check3.append(emothion['Confidence'])
            elif str(emothion['Type']) == 'FEAR':
                per4.append(str(emothion['Confidence']))
                per41.append(emothion['Confidence'])
                check4.append(emothion['Confidence'])
            elif str(emothion['Type']) == 'ANGRY':
                per5.append(str(emothion['Confidence']))
                per51.append(emothion['Confidence'])
                check5.append(emothion['Confidence'])
            elif str(emothion['Type']) == 'SAD':
                per6.append(str(emothion['Confidence']))
                per61.append(emothion['Confidence'])
                check6.append(emothion['Confidence'])
            elif str(emothion['Type']) == 'DISGUSTED':
                per7.append(str(emothion['Confidence']))
                per71.append(emothion['Confidence'])
                check7.append(emothion['Confidence'])
            elif str(emothion['Type']) == 'CALM':
                per8.append(str(emothion['Confidence']))
                per81.append(emothion['Confidence'])
                check8.append(emothion['Confidence'])
            if len(check1) == 5:
                c = list(range(8))
                c[0] = sum(check1)
                c[1] = sum(check2)
                c[2] = sum(check3)
                c[3] = sum(check4)
                c[4] = sum(check5)
                c[5] = sum(check6)
                c[6] = sum(check7)
                c[7] = sum(check8)
                if (c.index(max(c))) == 0:
                    per_result.append(0)
                elif (c.index(max(c))) == 1:
                    per_result.append(1)
                elif (c.index(max(c))) == 2:
                    per_result.append(2)
                elif (c.index(max(c))) == 3:
                    per_result.append(3)
                elif (c.index(max(c))) == 4:
                    per_result.append(4)
                elif (c.index(max(c))) == 5:
                    per_result.append(5)
                elif (c.index(max(c))) == 6:
                    per_result.append(6)
                elif (c.index(max(c))) == 7:
                    per_result.append(7)
                check1.clear()
                check2.clear()
                check3.clear()
                check4.clear()
                check5.clear()
                check6.clear()
                check7.clear()
                check8.clear()
                        
    # 結果をディスプレイに表示
    cv2.imshow('frame', frame)
    end = time.time()

video.release()
cap.release()
cv2.destroyAllWindows()

#csvファイルで保存
import pandas as pd
a = list(range(9))
a[0] = time
a[1] = per1
a[2] = per2
a[3] = per3
a[4] = per4
a[5] = per5
a[6] = per6
a[7] = per7
a[8] = per8
df = pd.DataFrame(a)
df.to_csv(hikensya+'_kanjo_'+junban+'.csv')

emotion = list(range(8))
emotion[0] = sum(per11)
emotion[1] = sum(per21)
emotion[2] = sum(per31)
emotion[3] = sum(per41)
emotion[4] = sum(per51)
emotion[5] = sum(per61)
emotion[6] = sum(per71)
emotion[7] = sum(per81)

s = ''
number = 0
sum_res = sum(emotion)
if (emotion.index(max(emotion))) == 0:
    s = 'HAPPY'
    number = 0
elif (emotion.index(max(emotion))) == 1:
    s = 'CONFUSED'
    number = 1
elif (emotion.index(max(emotion))) == 2:
    s = 'SURPRISED'
    number = 2
elif (emotion.index(max(emotion))) == 3:
    s = 'FEAR' 
    number = 3
elif (emotion.index(max(emotion))) == 4:
    s = 'ANGRY'
    number = 4
elif (emotion.index(max(emotion))) == 5:
    s = 'SAD'
    number = 5
elif (emotion.index(max(emotion))) == 6:
    s = 'DISGUSTED'
    number = 6
elif (emotion.index(max(emotion))) == 7:
    s = 'CALM'
    number = 7

print("HAPPY    :" + str(emotion[0]/sum_res))
print("CONFUSED :" + str(emotion[1]/sum_res))
print("SURPRISED:" + str(emotion[2]/sum_res))
print("FEAR     :" + str(emotion[3]/sum_res))
print("ANGRY    :" + str(emotion[4]/sum_res))
print("SAD      :" + str(emotion[5]/sum_res))
print("DISGUSTED:" + str(emotion[6]/sum_res))
print("CALM     :" + str(emotion[7]/sum_res))
print("Emotion  :" + s + "       Score   :" + str(emotion[number]/sum_res))
cap.release()
cv2.destroyAllWindows()

# 認識結果を表示するためのライブラリを読み込む
from matplotlib import pyplot as plt
from PIL import Image
import random
import numpy as np

#plot
import matplotlib
import matplotlib.pyplot as plt
#plt.plot("x","y","オプション(marker="記号", color = "色", linestyle = "線の種類")")
plt.plot(ti, per11, label='HAPPY', color = "magenta")
plt.plot(ti, per21, label='CONFUSED', color = "gold")
plt.plot(ti, per31, label='SURPRISED', color = "lime")
plt.plot(ti, per41, label='FEAR', color = "teal")
plt.plot(ti, per51, label='ANGRY', color = "green")
plt.plot(ti, per61, label='SAD', color = "blue")
plt.plot(ti, per71, label='DISGUSTED', color = "purple")
plt.plot(ti, per81, label='CALM', color = "orangered")
plt.legend(loc='upper left', fontsize=8) # (3)凡例表示
#メモリの設定
plt.xticks(np.arange(0, num, step=num/10 + 0.1))
plt.yticks(np.arange(0, 101, step=10))
#ラベルの設定
plt.xlabel("Time")
plt.ylabel("Confidence[%]")
#補助線の設定
plt.grid(True)
#保存・表示
plt.savefig(hikensya+"_aws_result_"+junban+".png")

plt.plot(per_result)

plt.yticks(np.arange(0, 4.1, step=1))
plt.yticks(np.arange(0, 7.1, step=1))

plt.savefig(hikensya+"_aws_result_kanjo_"+junban+".png")


count = list(range(8))
count[0] = per_result.count(0)
count[1] = per_result.count(1)
count[2] = per_result.count(2)
count[3] = per_result.count(3)
count[4] = per_result.count(4)
count[5] = per_result.count(5)
count[6] = per_result.count(6)
count[7] = per_result.count(7)
sum_c = len(per_result)
s1 = ''

if (count.index(max(count))) == 0:
    s1 = 'HAPPY'
    score = count[0]/sum_c * 100
elif (count.index(max(count))) == 1:
    s1 = 'CONFUSED'
    score = count[1]/sum_c * 100
elif (count.index(max(count))) == 2:
    s1 = 'SURPRISED'
    score = count[2]/sum_c * 100
elif (count.index(max(count))) == 3:
    s1 = 'FEAR' 
    score = count[3]/sum_c * 100
elif (count.index(max(count))) == 4:
    s1 = 'ANGRY'
    score = count[4]/sum_c * 100
elif (count.index(max(count))) == 5:
    s1 = 'SAD'
    score = count[5]/sum_c * 100
elif (count.index(max(count))) == 6:
    s1 = 'DISGUSTED'
    score = count[6]/sum_c * 100
elif (count.index(max(count))) == 7:
    s1 = 'CALM'
    score = count[7]/sum_c * 100
print("Emotion  :" + s1 + "       Score   :" + str(score))

print(str(emotion[0]/sum_res * 100))
print(str(emotion[1]/sum_res * 100))
print(str(emotion[2]/sum_res * 100))
print(str(emotion[3]/sum_res * 100))
print(str(emotion[4]/sum_res * 100))
print(str(emotion[5]/sum_res * 100))
print(str(emotion[6]/sum_res * 100))
print(str(emotion[7]/sum_res * 100))
print(s)
print(str(emotion[number]/sum_res * 100))

print(str(count[0]/sum_c * 100))
print(str(count[1]/sum_c * 100))
print(str(count[2]/sum_c * 100))
print(str(count[3]/sum_c * 100))
print(str(count[4]/sum_c * 100))
print(str(count[5]/sum_c * 100))
print(str(count[6]/sum_c * 100))
print(str(count[7]/sum_c * 100))
print(s1)
print(str(score))