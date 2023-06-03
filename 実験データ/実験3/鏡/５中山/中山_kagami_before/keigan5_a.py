from socket import socket, AF_INET, SOCK_STREAM


HOST        = 'localhost'
PORT        = 51000
MAX_MESSAGE = 2048
NUM_THREAD  = 4

CHR_CAN     = '\18'
CHR_EOT     = '\04'

import cv2

photos=[
    'happy.png',
    'angry.png',
    'surprised.png',
    'sad.png',
    'calm.png'
]

def com_receive():
    message = ["6S"]
    # 通信の確立
    sock = socket(AF_INET, SOCK_STREAM)
    sock.bind    ((HOST, PORT))
    sock.listen  (NUM_THREAD)
    print ('receiver ready, NUM_THREAD  = ' + str(NUM_THREAD))

    # メッセージ受信ループ
    while True:
        try:
            conn, addr = sock.accept()
            mess       = conn.recv(MAX_MESSAGE).decode('utf-8')
            conn.close()

            # 終了要求？
            if (mess == CHR_EOT):
                break

            # キャンセル？
            if (mess == CHR_CAN):
                print('cancel')
                continue

            # テキスト
            print ('message:' + mess)
            message.append(mess)
            
            img = cv2.imread(photos[int(mess)-1])
            
            # 画像の表示
            cv2.imshow("Image", img)
            
            cv2.waitKey(1000)
            
            #if mess == "Happy":
            if mess == "1": 
                dev1_move(180)
                
                dev2.move_to_pos(utils.deg2rad(30))
                sleep(0.3)
                dev2.move_to_pos(utils.deg2rad(-30))
                sleep(0.3)
                dev2.move_to_pos(utils.deg2rad(30))
                sleep(0.3)
            #elif  mess == "Angry & Confussed & Disgusted":
            elif mess == "2":
                dev1_move(0)
                dev2.move_to_pos(utils.deg2rad(0))
                sleep(0.2)
            #elif mess == "Surprised & Fear":
            elif mess == "3": 
                dev1_move(150)
                dev2.move_to_pos(utils.deg2rad(0))
                sleep(0.2)
            #elif mess == "Sad":
            elif mess == "4":
                dev1_move(45)
                dev2.move_to_pos(utils.deg2rad(60))
                sleep(0.4)
                dev2.move_to_pos(utils.deg2rad(-60))
                sleep(0.4)
            #elif mess == "Calm":
            elif mess == "5":
                dev1_move(90)
                dev2.move_to_pos(utils.deg2rad(0))
                sleep(0.2)
            elif mess == "exit":
                break
            
        except:
            print ('Error:' + mess)

    # 通信の終了
    sock.close()
    print ('end of receiver')
    
def proc():
    com_receive()
    
import sys
import os
import pathlib
from time import sleep
from concurrent.futures import ThreadPoolExecutor

current_dir = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir) + '/../') # give 1st priority to the directory where pykeigan exists

from pykeigan import blecontroller
from pykeigan import utils

"""
----------------------
モーターに接続し、各種情報の取得
----------------------
"""

def get_motor_informations():
    while True:
        if dev:
            print("\033[3;2H\033[2K")
            sys.stdout.flush()
            print('status {} '.format(dev.read_status()))
            sys.stdout.flush()

            print("\033[8;2H\033[2K")
            sys.stdout.flush()
            print('measurement {} '.format(dev.read_motor_measurement()))
            sys.stdout.flush()

            print("\033[12;2H\033[2K")
            sys.stdout.flush()
            print('imu_measurement {} '.format(dev.read_imu_measurement()))
            sys.stdout.flush()
        sleep(0.5)

"""
----------------------
回転情報
----------------------
"""
##モーター回転情報callback
def on_motor_measurement_cb(measurement):
    print("\033[2;2H\033[2K")
    print('measurement {} '.format(measurement))
    sys.stdout.flush()

##ログ情報callback
def on_motor_log_cb(log):
    print("\033[5;2H\033[2K")
    sys.stdout.flush()
    print('log {} '.format(log))
    sys.stdout.flush()
    
"""
----------------------
モータ接続
----------------------
"""
#motor1
dev1=blecontroller.BLEController("c3:fa:c9:a0:83:da")   #モーターのMACアドレス 参照 ble-simple-connection.py
dev1.enable_action()                                    #安全装置。初めてモーターを動作させる場合に必ず必要。
dev1.set_speed(utils.rpm2rad_per_sec(5000))                #rpm -> radian/sec      回転数
dev1.preset_position(0)                                 #現在位置の座標を0に設定
dev1.set_curve_type(0)

#motor2
dev2=blecontroller.BLEController("ca:08:5c:f5:0a:6d")   #モーターのMACアドレス 参照 ble-simple-connection.py
dev2.enable_action()                                    #安全装置。初めてモーターを動作させる場合に必ず必要。
dev2.set_speed(utils.rpm2rad_per_sec(5000))                #rpm -> radian/sec      回転数
dev2.preset_position(0)                                 #現在位置の座標を0に設定
dev2.set_curve_type(0)

"""
-------------
Move Fuction
-------------
"""

def dev1_move(num):
    dev1.move_to_pos(utils.deg2rad(num))
    sleep(0.5)

def reset():
    dev1.move_to_pos(utils.deg2rad(0),(utils.deg2rad(1000)/2))
    dev2.move_to_pos(utils.deg2rad(0),(utils.deg2rad(1000)/2))
    sleep(0.5)

reset()

import time

st = time.time()
proc()

while 0.5 < (time.time()-st) <= 2:
    proc()
    st = time.time()
    
reset()

while True:
    print("\033[18;2H")
    sys.stdout.flush()
    print("---------------------------------------")
    if sys.version_info<(3,0):
        inp = raw_input('Exit:[key input] >>')
    else:
        inp = input('Exit:[key input] >>')
    if inp !=None:
        dev1.set_led(1, 100, 100, 100)
        dev1.set_curve_type(1)
        dev1.disable_continual_imu_measurement()
        dev1.disconnect()
        dev2.set_led(1, 100, 100, 100)
        dev2.set_curve_type(1)
        dev2.disable_continual_imu_measurement()
        dev2.disconnect()
        break