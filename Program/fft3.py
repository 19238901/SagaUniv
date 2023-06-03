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
frep = []
amp1 = []
num_fft = []
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
        
        # Wave figure
        wave_figure = data[self.START:self.START + self.N]
        # Wave time
        wave_time = range(self.START, self.START + self.N)
        # Frequency
        freqlist = np.fft.fftfreq(self.N, d = 1.0 / self.RATE) 
        # Spectrum power
        x = np.fft.fft(data[self.START:self.START + self.N])
        
        amplitudeSpectrum = [np.sqrt(c.real ** 2 + c.imag ** 2) for c in x]
        amp1.append(max(amplitudeSpectrum))
        
        frep_n = amplitudeSpectrum.index(max(amplitudeSpectrum))
        if freqlist[frep_n] < 0:
            freqlist[frep_n] = (-1)*freqlist[frep_n]
        frep.append(freqlist[frep_n])
        num_fft.append(len(frep))
        # Plot setdata
        self.curve_wave.setData(wave_time, wave_figure)
        self.curve_spectrum.setData(freqlist, amplitudeSpectrum)
        #csvファイルで保存
        import pandas as pd
        a = list(range(3))
        a[0] = num_fft
        a[1] = frep #周波数
        a[2] = amp1 #振幅
        df = pd.DataFrame(a)
        df.to_csv('fft31_kanjo.csv')

    def audioinput(self):
        ret = self.stream.read(self.CHUNK ,exception_on_overflow=False )
        ret = np.fromstring(ret, np.float32)

        return ret

if __name__ == "__main__":
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
    ax1.set_xlabel("time")  #x軸ラベル
    ax1.set_ylabel("Frequency[Hz]")  #y軸ラベル
    ax2.set_ylabel("Amplitude spectrum[f]")  #y2軸ラベル
    
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2 ,loc='lower right')
    
    #メモリの設定
    plt.xticks(np.arange(0, len(num_fft), step=50))
    ax1.set_yticks(np.arange(0, 8000, step=1000))
    ax2.set_yticks(np.arange(0, max(amp1), step=50))

    #補助線の設定
    plt.grid(True)
    
    #保存・表示
    plt.savefig("fft_result.jpg")
    plt.show()