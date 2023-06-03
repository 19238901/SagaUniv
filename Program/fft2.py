#!/usr/bin/env python
# encoding: utf-8

## Module infomation ###
# Python (3.4.4)
# numpy (1.10.2)
# PyAudio (0.2.9)
# pyqtgraph (0.9.10)
# PyQt4 (4.11.4)
# All 32bit edition
########################

import pyaudio
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
import math
import pandas as pd
from scipy.stats import norm
import time

data_f = pd.DataFrame([344.11337209302326, 333.3333333333333, 346.875])
data_a = pd.DataFrame([51.180272392110105, 44.67819497722985, 37.36864426768469])

p_var_f = 51.20209341  # 母分散
s_mean_f = data_f.mean()  # 標本平均
n_f = len(data_f)  # 標本数

p_var_a = 47.74460229  # 母分散
s_mean_a = data_a.mean()  # 標本平均
n_a = len(data_a)  # 標本数

bottom_f, up_f = norm.interval(0.95, loc=s_mean_f, scale=math.sqrt(p_var_f/n_f))
bottom_a, up_a = norm.interval(0.95, loc=s_mean_a, scale=math.sqrt(p_var_a/n_a))

frep = []
amp1 = []
num_fft = []
motion = []
result = []
check1 = []
start_time = time.time()
ex_num = '';
import time
class SpectrumAnalyzer:
    FORMAT = pyaudio.paFloat32
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1024
    START = 0
    N = 1024
    WAVE_RANGE = 1
    SPECTRUM_RANGE = 50
    UPDATE_SECOND = 10

    def __init__(self):
        #Pyaudio Configuration
        self.pa = pyaudio.PyAudio()
        self.stream = self.pa.open(format = self.FORMAT,
                channels = self.CHANNELS,
                rate = self.RATE,
                input = True,
                output = False,
                frames_per_buffer = self.CHUNK)
       
        # Graph configuration
        # Application
        self.app = QtGui.QApplication([])
        self.app.quitOnLastWindowClosed()
        # Window
        self.win = QtGui.QMainWindow()
        self.win.setWindowTitle("SpectrumAnalyzer")
        self.win.resize(800, 600)
        self.centralwid = QtGui.QWidget()
        self.win.setCentralWidget(self.centralwid) 
        # Layout
        self.lay = QtGui.QVBoxLayout()
        self.centralwid.setLayout(self.lay)
        # Wave figure window setting
        self.plotwid1 = pg.PlotWidget(name="wave")
        self.plotitem1 = self.plotwid1.getPlotItem()
        self.plotitem1.setMouseEnabled(x = False, y = False) 
        self.plotitem1.setYRange(self.WAVE_RANGE * -1, self.WAVE_RANGE * 1)
        self.plotitem1.setXRange(self.START, self.START + self.N, padding = 0)
        # Spectrum windows setting
        self.plotwid2 = pg.PlotWidget(name="spectrum")
        self.plotitem2 = self.plotwid2.getPlotItem()
        self.plotitem2.setMouseEnabled(x = False, y = False) 
        self.plotitem2.setYRange(0, self.SPECTRUM_RANGE)
        self.plotitem2.setXRange(0, self.RATE / 2, padding = 0)
        # Wave figure Axis
        self.specAxis1 = self.plotitem1.getAxis("bottom")
        self.specAxis1.setLabel("Time [sample]")
        # Spectrum Axis
        self.specAxis2 = self.plotitem2.getAxis("bottom")
        self.specAxis2.setLabel("Frequency [Hz]")
        #Plot data
        self.curve_wave = self.plotitem1.plot()
        self.curve_spectrum = self.plotitem2.plot()
        #Widget
        self.lay.addWidget(self.plotwid1)
        self.lay.addWidget(self.plotwid2)
        #Show plot window
        self.win.show()
        #Update timer setting
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(self.UPDATE_SECOND)

    def update(self):
        #Get audio input
        data = self.audioinput()
        motion_n = 0
        # Wave figure
        wave_figure = data[self.START:self.START + self.N]
        # Wave time
        wave_time = range(self.START, self.START + self.N)
        # Frequency
        freqlist = np.fft.fftfreq(self.N, d = 1.0 / self.RATE) 
        # Spectrum power
        x = np.fft.fft(data[self.START:self.START + self.N])
        
        amplitudeSpectrum = [np.sqrt(c.real ** 2 + c.imag ** 2) for c in x]
        
        frep_n = amplitudeSpectrum.index(max(amplitudeSpectrum))
        
        end = time.time()
        
        if freqlist[frep_n] < 0:
            freqlist[frep_n] = (-1)*freqlist[frep_n]
        if 2000 >= freqlist[frep_n] and  freqlist[frep_n] >= 87 and max(amplitudeSpectrum) > 30:
            if bottom_f > freqlist[frep_n]:
                motion_n = -1
            elif up_f < freqlist[frep_n]:
                motion_n = 1
            else:
                motion_n = 0
            if bottom_a > max(amplitudeSpectrum):
                motion_n = motion_n * 0.75
            elif up_a < max(amplitudeSpectrum):
                motion_n = motion_n * 2
            else:
                motion_n = motion_n * 1
                    
            frep.append(freqlist[frep_n])
            amp1.append(max(amplitudeSpectrum))
            num_fft.append(len(frep))
            motion.append(motion_n)
            check1.append(motion_n)
                
            if len(check1) == 5:
                averrage_5 = sum(check1)/5
                if -2 <= averrage_5 < -1:
                    result.append(0)
                elif -1 <= averrage_5 < 0:
                    result.append(1)
                elif 0 <= averrage_5 <= 1:
                    result.append(2)
                elif 1 < averrage_5 <= 2:
                    result.append(3)
                check1.clear()
                    
        # Plot setdata
        self.curve_wave.setData(wave_time, wave_figure)
        self.curve_spectrum.setData(freqlist, amplitudeSpectrum)
        #csvファイルで保存
        import pandas as pd
        a = list(range(4))
        a[0] = num_fft
        a[1] = frep #周波数 
        a[2] = amp1 #振幅
        a[3] = motion #感情
        df = pd.DataFrame(a)
        file_text="fft31_kanjo.csv"
        df.to_csv(file_text)
    
    def audioinput(self):
        ret = self.stream.read(self.CHUNK ,exception_on_overflow=False )
        ret = np.fromstring(ret, np.float32)

        return ret

if __name__ == "__main__":
    start_time = time.time()
    
    spec = SpectrumAnalyzer()
    QtGui.QApplication.instance().exec_()
    # 認識結果を表示するためのライブラリを読み込む
    
    import matplotlib.cm as cm
    from PIL import Image
    import random
    import numpy as np
    
    #plot
    import matplotlib
    import matplotlib.pyplot as plt
    
    #plt.plot("x","y","オプション(marker="記号", color = "色", linestyle = "線の種類")")
    fig = plt.figure()
    ax1 = fig.add_subplot()
    
    ax2 = ax1.twinx()
    
    ax1.plot(num_fft, frep,color = "blue", label = "Frequency")
    ax2.plot(num_fft, amp1,color = "red", label = "Amplitude spectrum")
    ax1.set_title("Statistical data")    #タイトル
    ax1.set_xlabel("Time")  #x軸ラベル
    ax1.set_ylabel("Frequency[Hz]")  #y軸ラベル
    ax2.set_ylabel("Amplitude spectrum[f]")  #y2軸ラベル
    
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2 ,loc='lower right')
    
    #メモリの設定
    plt.xticks(np.arange(0, len(num_fft)+1, step=len(num_fft)/10))
    ax1.set_yticks(np.arange(0, 1500, step=500))
    ax2.set_yticks(np.arange(15, max(amp1)+1, step=max(amp1)/10))

    #補助線の設定
    plt.grid(True)
    
    #保存・表示
    plt.savefig("fft_result.jpg")
    plt.show()
    
    plt.plot(num_fft, motion)
    #メモリの設定
    plt.xticks(np.arange(0, len(num_fft), step=len(num_fft)/10))
    plt.yticks(np.arange(-2, 2.1, step=1))
    #ラベルの設定
    plt.xlabel("Time")
    plt.ylabel("Emotion")
    #補助線の設定
    plt.grid(True)
    #保存・表示
    plt.savefig("motion_fft_result.png")
    plt.show()
    
    print("Ave_fft:" + str(sum(frep)/len(frep)) + "[Hz]")
    print("Ave_amp:" + str(sum(amp1)/len(amp1)) + "[F]")
    print("Emotion:" + str(sum(motion)/len(motion)))
    E = ''
    if 1 < sum(motion)/len(motion) <= 2:
        E = 'STRONG POSITIVE'
    elif 0 <= sum(motion)/len(motion) <= 1:
        E = 'WEAK POSITIVE'
    elif -1 <= sum(motion)/len(motion) < 0:
        E = 'WEAK NEGATIVE'
    elif -2 <= sum(motion)/len(motion) < -1:
        E = 'STRONG NEGATIVE'
    print("Emotion:" + E)
    
    
    count = list(range(4))
    count[0] = result.count(0)
    count[1] = result.count(1)
    count[2] = result.count(2)
    count[3] = result.count(3)
    sum_c = len(result)
    s_res =""
    print("Strong Negative  :" + str(count[0]/sum_c * 100))
    print("Weak Negative    :" + str(count[1]/sum_c * 100))
    print("Weak Positive    :" + str(count[2]/sum_c * 100))
    print("Strong Positive  :" + str(count[3]/sum_c * 100))
    score = 0
    if (count.index(max(count))) == 0:
        print("Emotion : Strong Negative       score :" + str(count[0]/sum_c * 100))
        score = count[0]/sum_c * 100
        s_res = "Strong Negative"
    elif (count.index(max(count))) == 1:
        print("Emotion : Weak Negative         score :" + str(count[1]/sum_c * 100))
        s_res = "Weak Negative"
        score = count[1]/sum_c * 100
    elif (count.index(max(count))) == 2:
        print("Emotion : Weak Positive         score :" + str(count[2]/sum_c * 100))
        s_res = "Weak Positive"
        score = count[2]/sum_c * 100
    elif (count.index(max(count))) == 3:
        print("Emotion : Strong Negative       score :" + str(count[3]/sum_c * 100))
        s_res = "Strong Positive"
        score = count[3]/sum_c * 100
    print(str(sum(frep)/len(frep)))
    print(str(sum(amp1)/len(amp1)))
    print(E)
    print(str((sum(motion)/len(motion))))
    print(str(count[0]/sum_c * 100))
    print(str(count[1]/sum_c * 100))
    print(str(count[2]/sum_c * 100))
    print(str(count[3]/sum_c * 100))
    print(s_res)
    print(str(score))