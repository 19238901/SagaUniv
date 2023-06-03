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
junban = input()
import time

import matplotlib.pyplot as plt
import numpy as np

# 描画領域を取得
fig, ax = plt.subplots(1, 1)
# y軸方向の描画幅を指定
ax.set_ylim((0, 1))
# x軸:時刻
x = np.arange(0, 30, 0.5)

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
name = "video_" + junban + ".mp4"#保存名　video + 現在時刻 + .mp4
video = cv2.VideoWriter(name, fourcc, fps, (w,h))#(保存名前、fourcc,fps,サイズ)

class realtime_plot(object):

    def __init__(self):
        self.fig = plt.figure(figsize=(12,8))
        self.initialize()

    def initialize(self):
        self.fig.suptitle('monitoring sample', size=12)
        self.fig.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.90, wspace=0.2, hspace=0.6)
        self.ax00 = plt.subplot2grid((2,2),(0,0))
        #self.ax10 = plt.subplot2grid((2,2),(1,0))
        #self.ax01 = plt.subplot2grid((2,2),(0,1))
        #self.ax11 = plt.subplot2grid((2,2),(1,1))
        self.ax00.grid(True)
        #self.ax10.grid(True)
        #self.ax01.grid(True)
        #self.ax11.grid(True)
        self.ax00.set_title('real time result')
        #self.ax10.set_title('histogram')
        #self.ax01.set_title('correlation')
        #self.ax11.set_title('optimized result')
        self.ax00.set_xlabel('Count')
        self.ax00.set_ylabel('Confidence[%]')
        #self.ax01.set_xlabel('correct')
        #self.ax01.set_ylabel('predict')
        #self.ax11.set_xlabel('correct')
        #self.ax11.set_ylabel('predict')

        # プロットの初期化
        self.lines000, = self.ax00.plot([-1,-1],[1,1],label='HAPPY', color = "magenta")
        self.lines001, = self.ax00.plot([-1,-1],[1,1],label='CONFUDSED', color = "gold")
        self.lines002, = self.ax00.plot([-1,-1],[1,1],label='SURPRISED', color = "lime")
        self.lines003, = self.ax00.plot([-1,-1],[1,1],label='FEAR', color = "teal")
        self.lines004, = self.ax00.plot([-1,-1],[1,1],label='ANGRY', color = "green")
        self.lines005, = self.ax00.plot([-1,-1],[1,1],label='SAD', color = "blue")
        self.lines006, = self.ax00.plot([-1,-1],[1,1],label='DISGUSTED', color = "purple")
        self.lines007, = self.ax00.plot([-1,-1],[1,1],label='CALM', color = "orangered")
        
        #self.lines100 = self.ax10.hist([-1,-1],label='res1')
        #self.lines101 = self.ax10.hist([-1,-1],label='res2')
        #self.lines01, = self.ax01.plot([-1,-1],[1,1],'.')
        #self.lines11, = self.ax11.plot([-1,-1],[1,1],'.r')

    # 値名と値を代入した辞書タイプのdataから，必要なデータをsubplotの値に代入します
    def set_data(self,data):

        self.lines000.set_data(data['ti'],data['per11'])
        self.lines001.set_data(data['ti'],data['per21'])
        self.lines002.set_data(data['ti'],data['per31'])
        self.lines003.set_data(data['ti'],data['per41'])
        self.lines004.set_data(data['ti'],data['per51'])
        self.lines005.set_data(data['ti'],data['per61'])
        self.lines006.set_data(data['ti'],data['per71'])
        self.lines007.set_data(data['ti'],data['per81'])
        self.ax00.set_xlim((0,len(data['ti'])))
        self.ax00.set_ylim((0,100))
        # 凡例を固定するために必要
        self.ax00.legend(loc='upper left')

        #self.lines01.set_data(data['corr'],data['pred'])
        #self.ax01.set_xlim((-2,12))
        #self.ax01.set_ylim((-2,12))

        # ヒストグラムはset_dataがないので，更新毎に新たに作り直します
        #self.ax10.cla()
        #self.ax10.set_title('histogram')
        #self.ax10.grid(True)
       # self.lines100 = self.ax10.hist(data['corr'],label='corr')
        #self.lines101 = self.ax10.hist(data['pred'],label='pred')
        #self.ax10.set_xlim((-0.5,9.5))
        #self.ax10.set_ylim((0,20))
        #self.ax10.legend(loc='upper right')

        # タイトルやテキストを更新する場合，更新前の値がfigureに残ったままになるので，更新毎に新たに作り直します
        #bbox_props = dict(boxstyle='square,pad=0.3',fc='gray')
        #self.ax11.cla()
        #self.ax11.grid(True)
        #self.ax11.set_xlabel('correct')
        #self.ax11.set_ylabel('predict')
        #self.ax11.set_title('optimized result')
        #self.ax11.text(-1.5,10.5,data['text'], ha='left', va='center',size=11,bbox=bbox_props)
        #self.lines = self.ax11.plot(data['opt_corr'],data['opt_pred'],'.')
        #self.ax11.set_xlim((-2,12))
        #self.ax11.set_ylim((-2,12))

    def pause(self,second):
        plt.pause(second)

# 使用例
RP = realtime_plot()
data = {}
opt_coef = 0

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
    data['ti'] = ti
    data['per11'] = per11
    data['per21'] = per21
    data['per31'] = per31
    data['per41'] = per41
    data['per51'] = per51
    data['per61'] = per61
    data['per71'] = per71
    data['per81'] = per81

    RP.set_data(data)
    RP.pause(0.1)
    end = time.time()

video.release()
cap.release()
cv2.destroyAllWindows()

#csvファイルで保存
import pandas as pd
a = list(range(9))
a[0] = Dtime
a[1] = per1
a[2] = per2
a[3] = per3
a[4] = per4
a[5] = per5
a[6] = per6
a[7] = per7
a[8] = per8
df = pd.DataFrame(a)
df.to_csv('kanjo_'+junban+'.csv')

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
plt.xticks(np.arange(0, num, step=num/10))
plt.yticks(np.arange(0, 101, step=10))
#ラベルの設定
plt.xlabel("Time")
plt.ylabel("Confidence[%]")
#補助線の設定
plt.grid(True)
#保存・表示
plt.savefig("aws_result_"+junban+".png")
plt.show()

sum_per = [sum(per11),sum(per21),sum(per31),sum(per41),sum(per51),sum(per61),sum(per71),sum(per81)]
label_per = ['HAPPY','CONFUSED','SURPRISED','FEAR','ANGRY','SAD','DISGUSTED','CALM']
colors = ["magenta","gold","lime","teal","green","blue","purple","orangered"]
plt.pie(sum_per, labels = label_per ,colors=colors, wedgeprops={'linewidth': 3, 'edgecolor':"white"})
plt.savefig("aws_result_en_"+junban+".png")
plt.show()

plt.plot(per_result)

plt.yticks(np.arange(0, 4.1, step=1))
plt.yticks(np.arange(0, 7.1, step=1))

plt.savefig("aws_result_kanjo_"+junban+".png")
plt.show()

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