from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QHBoxLayout,QVBoxLayout, QGridLayout,QPushButton,QDesktopWidget,QScrollArea
from PyQt5.QtGui import QPixmap, QPalette
import sys
import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread, QSize
import numpy as np
import mediapipe as mp
import tensorflow 
from tensorflow import keras
import pyautogui  as pai
from keras.models import load_model
import keyboard
from enum import IntEnum

import math

pai.FAILSAFE = False
global self_thread
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
background = None


fList = []
image_width = 512
image_height = 513

model1 = load_model("final_hand_model.h5")
fList = []
CATEGORIES =  ["1","2","3","4","5","6","7","8","9","0"]
Categories = ['A','B','C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 
           'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 
           'W', 'X', 'Y', 'Z', 'Nothing', 'del', 'Space']


class Gest(IntEnum):
  
    FIST = 0
    PINKY = 1
    RING = 2
    MID = 4
    LAST3 = 7
    INDEX = 8
    FIRST2 = 12
    FIRST3 = 14
    LAST4 = 15
    THUMB = 16    
    PALM = 31

    V_GEST = 33
    TWO_FINGER_CLOSED = 34
 
class HandRecongition:
    def __init__(self):
        self.finger = 0
        self.ori_gesture = Gest.PALM
        self.prev_gesture = Gest.PALM
        self.frame_count = 0
        self.hand_result = None
        

    def update_hand_result(self, hand_result):
        self.hand_result = hand_result
    
    def get_signed_dist(self, point):
       
        sign = -1
        if self.hand_result.landmark[point[0]].y < self.hand_result.landmark[point[1]].y:
            sign = 1
        dist = (self.hand_result.landmark[point[0]].x - self.hand_result.landmark[point[1]].x)**2
        dist += (self.hand_result.landmark[point[0]].y - self.hand_result.landmark[point[1]].y)**2
        dist = math.sqrt(dist)
        return dist*sign

    def get_dist(self, point):
    
      
        dist = (self.hand_result.landmark[point[0]].x - self.hand_result.landmark[point[1]].x)**2
        dist += (self.hand_result.landmark[point[0]].y - self.hand_result.landmark[point[1]].y)**2
        dist = math.sqrt(dist)
        return dist
    
    def set_finger_state(self):
   
        if self.hand_result == None:
            return

        points = [[8,5,0],[12,9,0],[16,13,0],[20,17,0]]
        self.finger = 0
        self.finger = self.finger | 0 
        for idx,point in enumerate(points):
            
            dist = self.get_signed_dist(point[:2])
            dist2 = self.get_signed_dist(point[1:])
            
            try:
                ratio = round(dist/dist2,1)
            except:
                ratio = round(dist/0.01,1)

            self.finger = self.finger << 1
            if ratio > 0.5 :
                self.finger = self.finger | 1
        

    

    def get_gesture(self):
     
        if self.hand_result == None:
            return Gest.PALM
        current_gesture = Gest.PALM
        
    
        
        if Gest.FIRST2 == self.finger :
            
                current_gesture = Gest.V_GEST
            
        else:
                current_gesture =  self.finger
        
        if current_gesture == self.prev_gesture:
            self.frame_count += 1
        else:
            self.frame_count = 0

        self.prev_gesture = current_gesture

        if self.frame_count > 4 :
            self.ori_gesture = current_gesture
        return self.ori_gesture
        

class Controller:


    tx_old = 0
    ty_old = 0
    flag = False
    grabflag = False
    prev_hand = None     
    
    def get_position(hand_result):
    
        point = 5
        position = [hand_result.landmark[point].x ,hand_result.landmark[point].y]
        sx,sy = pai.size()
        x_old,y_old = pai.position()
        x = int(position[0]*sx)
        y = int(position[1]*sy)
        if Controller.prev_hand is None:
            Controller.prev_hand = x,y
        delta_x = x - Controller.prev_hand[0]
        delta_y = y - Controller.prev_hand[1]

        distsq = delta_x**2 + delta_y**2
        ratio = 1
        Controller.prev_hand = [x,y]

        if distsq <= 25:
            ratio = 0
        elif distsq <= 900:
            ratio = 0.07 * (distsq ** (1/2))
        else:
            ratio = 2.1
        x , y = x_old + delta_x*ratio , y_old + delta_y*ratio
        return (x,y)


    def handle_controls(gesture, hand_gesture):  
             
        x,y = None,None
        if gesture != Gest.PALM :
            x,y = Controller.get_position(hand_gesture)
        
        
        if gesture != Gest.FIST and Controller.grabflag:
            Controller.grabflag = False
            pai.mouseUp(button = "left")

       
        if gesture == Gest.V_GEST:
            Controller.flag = True
            pai.moveTo(x, y, duration = 0.1)
        
        elif gesture == Gest.FIST:
            if not Controller.grabflag : 
                Controller.grabflag = True
                pai.mouseDown(button = "left")
            pai.moveTo(x, y, duration = 0.1)
        elif gesture == Gest.PINKY:
            pai.scroll(120)
        elif gesture == Gest.LAST3:
            pai.scroll(-120)
        elif gesture == Gest.LAST4 and Controller.flag:
            pai.click()
            Controller.flag = False

        elif gesture == Gest.INDEX and Controller.flag:
            pai.click(button='right')
            Controller.flag = False

        elif gesture == Gest.FIRST3 and Controller.flag:
            pai.doubleClick()
            Controller.flag = False


class VideoThread(QThread):
 
    change_pixmap_signal = pyqtSignal(np.ndarray)
    cap = None
    CAM_HEIGHT = None
    CAM_WIDTH = None
    hr_major = None 
    hr_minor = None 
    dom_hand = True

    def __init__(self):
        super().__init__()
        self._run_flag = True
        
        VideoThread.cap = cv2.VideoCapture(0)
        VideoThread.CAM_HEIGHT = VideoThread.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        VideoThread.CAM_WIDTH = VideoThread.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    
    def prepare(self,frame):
        IMG_SIZE = 128
        minValue = 70
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(5,5),2)
        th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
        ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        resized=cv2.resize(res,(IMG_SIZE,IMG_SIZE))
        
        reshaped=np.reshape(resized,(1,IMG_SIZE,IMG_SIZE,1))
        
        return reshaped.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


 
              
    def run(self):
        nof = 1
        fc = 1 # Frame count for guide animation
        
        roi_top = 20
        roi_bottom = 300
        roi_right = 300
        roi_left = 600
        global self_thread
        i=0
        while 1:
            if(self_thread ==1):
                with mp_hands.Hands(max_num_hands = 2,min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
                    while VideoThread.cap.isOpened() :
                        nof = nof + 1
                        check, frame = VideoThread.cap.read()
                        image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
                        image.flags.writeable = False
                        results = hands.process(image)
                        
                        image.flags.writeable = True
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        fingercount  = 0
                        frame = cv2.flip(frame,1)
                        frame_copy = frame
                        roi = frame[roi_top:roi_bottom, roi_right:roi_left]
                        IMG_SIZE = 64
                        if results.multi_hand_landmarks:

                            for hand_landmarks in results.multi_hand_landmarks:
                                handidx = results.multi_hand_landmarks.index(hand_landmarks)
                                handlabel = results.multi_handedness[handidx].classification[0].label

                                handlandmarks = []

                                for lm in hand_landmarks.landmark :
                                    handlandmarks.append([lm.x, lm.y])
                                if( handlabel == "Left" and handlandmarks[4][0] > handlandmarks[3][0]):
                                    fingercount = fingercount + 1
                                elif ( handlabel == "Right" and handlandmarks[4][0] < handlandmarks[3][0]):
                                    fingercount = fingercount + 1
                                
                                if handlandmarks[8][1] < handlandmarks[6][1]:
                                    fingercount = fingercount + 1
                                if handlandmarks[12][1] < handlandmarks[10][1]:
                                    fingercount = fingercount + 1
                                    
                                if handlandmarks[16][1] < handlandmarks[14][1]:
                                    fingercount = fingercount + 1
                                if handlandmarks[20][1] < handlandmarks[18][1]:
                                    fingercount = fingercount + 1
                                if(fingercount == 10):
                                        self_thread = 2
                                        break
                                
                                IMG_SIZE = 128
                                minValue = 70
                                gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
                                blur = cv2.GaussianBlur(gray,(5,5),2)
                                th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
                                ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
                                resized=cv2.resize(res,(IMG_SIZE,IMG_SIZE))
                                
                                reshaped=np.reshape(resized,(1,IMG_SIZE,IMG_SIZE,1))
                                
                                reshaped.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
                                
                                prediction = model1.predict(reshaped)
                                pList = prediction.tolist()
                                predictClass = Categories[pList[0].index(max(pList[0]))]
                                fList.append(str(predictClass))
                                cv2.putText(frame_copy, predictClass, (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                                if len(fList) >= 10:
                                    for item in fList:
                                        if fList[0] != item:
                                            check = False
                                            break
                                    if(check == True):
                                        repeated = fList[0]
                                        fList.clear()
                                        
                                        if(repeated  != "Nothing"):
                                            if(repeated == "del"):
                                                keyboard.press('del')
                                            elif(repeated == "Space"):
                                                keyboard.write(' ')
                                            else:
                                                keyboard.write(repeated)
                                    else:
                                        fList.clear()
                                        

                        
                                cv2.rectangle(frame_copy, (roi_left, roi_top), (roi_right, roi_bottom), (0,0,255), 2)            
                                
                        self.change_pixmap_signal.emit(frame_copy)
                        if self_thread!= 1:
                                    break
                
            elif(self_thread == 2):
                handmajor = HandRecongition()
                with mp_hands.Hands(max_num_hands = 2,min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
                    while VideoThread.cap.isOpened():
                        success, image = VideoThread.cap.read()

                        if not success:
                            print("Ignoring empty camera frame.")
                            continue
                        
                        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                        image.flags.writeable = False
                        results = hands.process(image)
                        
                        image.flags.writeable = True
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                        if results.multi_hand_landmarks:
                            
                            for asd in results.multi_hand_landmarks:
                                handmajor.update_hand_result(asd)                   
                                handmajor.set_finger_state()
                                gest_name = handmajor.get_gesture()
                                Controller.handle_controls(gest_name, handmajor.hand_result)
                            
                                for hand_landmarks in results.multi_hand_landmarks:
                                    mpDraw.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()) 
                        else:
                            Controller.prev_hand = None

                        self.change_pixmap_signal.emit(image)
                       
                        
                        if self_thread !=2:
                            break
            elif(self_thread == 3):
                with mp_hands.Hands(max_num_hands = 2,min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
                    while VideoThread.cap.isOpened() :

                        success, image = VideoThread.cap.read()

                        if not success:
                            print("Ignoring empty camera frame.")
                            continue
                        
                        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                        image.flags.writeable = False
                        results = hands.process(image)
                        
                        image.flags.writeable = True
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        fingercount  = 0
                        check = True
                        if results.multi_hand_landmarks:

                            for hand_landmarks in results.multi_hand_landmarks:
                                handidx = results.multi_hand_landmarks.index(hand_landmarks)
                                handlabel = results.multi_handedness[handidx].classification[0].label

                                handlandmarks = []

                                for lm in hand_landmarks.landmark :
                                    handlandmarks.append([lm.x, lm.y])
                                if( handlabel == "Left" and handlandmarks[4][0] > handlandmarks[3][0]):
                                    fingercount = fingercount + 1
                                elif ( handlabel == "Right" and handlandmarks[4][0] < handlandmarks[3][0]):
                                    fingercount = fingercount + 1
                                
                                if handlandmarks[8][1] < handlandmarks[6][1]:
                                    fingercount = fingercount + 1
                                if handlandmarks[12][1] < handlandmarks[10][1]:
                                    fingercount = fingercount + 1
                                    
                                if handlandmarks[16][1] < handlandmarks[14][1]:
                                    fingercount = fingercount + 1
                                if handlandmarks[20][1] < handlandmarks[18][1]:
                                    fingercount = fingercount + 1
                                mpDraw.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()) 

                            fList.append(str(fingercount))    
                            cv2.putText(image, str(fingercount), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

                            if len(fList) >= 15:
                                for item in fList:
                                    if fList[0] != item:
                                        check = False
                                        break
                                
                                if(check == True):
                                    repeated = fList[0]
                                    fList.clear()

                                    if(repeated == '10'):
                                        self_thread = 2
                                        break
                                    else:
                                        keyboard.write(repeated)
                                else:
                                    fList.clear()
                                

                       
                        
                        self.change_pixmap_signal.emit(image)
                        if self_thread!= 3:
                                break
        
    def stop(self):

        VideoThread.cap.release()
        cv2.destroyAllWindows()
        self._run_flag = False
        exit()
class NumbersImage(QWidget):

    def __init__(self):
        super().__init__()
        self.resize(800,500)
        self.image_frame = QLabel(self)
        self.image_frame2 = QLabel(self)
        self.image_frame3 = QLabel(self)
        self.image_frame4 = QLabel(self)
        self.image_frame5 = QLabel(self)
        self.image_frame6 = QLabel(self)
        self.image_frame7 = QLabel(self)
        self.image_frame8 = QLabel(self)
        self.image_frame9 = QLabel(self)
        self.image_frame10 = QLabel(self)
        self.image_frame0 = QLabel(self)


        color = (0, 0, 255)
        org = (50, 50)
        fontScale = 3
        thickness = 7
        self.image = cv2.imread("Numbers_image\Sign 1 (60).jpeg")
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.image, '1', (10,350), font, fontScale, color, thickness, cv2.LINE_AA)
        self.image = cv2.resize(self.image, (200,200))
        self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.image_frame.setPixmap(QtGui.QPixmap.fromImage(self.image))
        
        

        self.image = cv2.imread("Numbers_image\Sign 2 (389).jpeg")
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.image, '2', (10,350), font, fontScale, color, thickness, cv2.LINE_AA)
        self.image = cv2.resize(self.image, (200,200))
        self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.image_frame2.setPixmap(QtGui.QPixmap.fromImage(self.image))
       
        

       
        
        self.image = cv2.imread("Numbers_image\Sign 3.jpeg")
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.image, '3', (10,350), font, fontScale, color, thickness, cv2.LINE_AA)
        self.image = cv2.resize(self.image, (200,200))
        self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.image_frame3.setPixmap(QtGui.QPixmap.fromImage(self.image))
      
        

        self.image = cv2.imread("Numbers_image\Sign 4 (2).jpeg")
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.image, '4', (10,350), font, fontScale, color, thickness, cv2.LINE_AA)
        self.image = cv2.resize(self.image, (200,200))
        self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.image_frame4.setPixmap(QtGui.QPixmap.fromImage(self.image))
        
        self.image = cv2.imread("Numbers_image\Sign 5 (31).jpeg")
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.image, '5', (10,350), font, fontScale, color, thickness, cv2.LINE_AA)
        self.image = cv2.resize(self.image, (200,200))
        self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.image_frame5.setPixmap(QtGui.QPixmap.fromImage(self.image))

        self.image = cv2.imread("Numbers_image\Sign6.png")
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.image, '6', (10,450), font, fontScale, color, thickness, cv2.LINE_AA)
        self.image = cv2.resize(self.image, (200,200))
        self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.image_frame6.setPixmap(QtGui.QPixmap.fromImage(self.image))

        self.image = cv2.imread("Numbers_image\Sign7.png")
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.image, '7', (10,350), font, fontScale, color, thickness, cv2.LINE_AA)
        self.image = cv2.resize(self.image, (200,200))
        self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.image_frame7.setPixmap(QtGui.QPixmap.fromImage(self.image))
      
        

        self.image = cv2.imread("Numbers_image\Sign8.png")
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.image, '8', (10,350), font, fontScale, color, thickness, cv2.LINE_AA)
        self.image = cv2.resize(self.image, (200,200))
        self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.image_frame8.setPixmap(QtGui.QPixmap.fromImage(self.image))
        
        self.image = cv2.imread("Numbers_image\Sign9.png")
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.image, '9', (10,350), font, fontScale, color, thickness, cv2.LINE_AA)
        self.image = cv2.resize(self.image, (200,200))
        self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.image_frame9.setPixmap(QtGui.QPixmap.fromImage(self.image))

        self.image = cv2.imread("Numbers_image\Sign 0 (13).jpeg")
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.image, '0', (10,350), font, fontScale, color, thickness, cv2.LINE_AA)
        self.image = cv2.resize(self.image, (200,200))
        self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.image_frame0.setPixmap(QtGui.QPixmap.fromImage(self.image))
        
        self.image = cv2.imread("Numbers_image\Sign10.png")
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.image, 'Close', (10,450), font, fontScale, color, thickness, cv2.LINE_AA)
        self.image = cv2.resize(self.image, (200,200))
        self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.image_frame10.setPixmap(QtGui.QPixmap.fromImage(self.image))
        self.layout = QHBoxLayout(self)
        self.scrollArea= QScrollArea()
        self.widget= QWidget()
        
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QWidget()
        
        
        
        gbox = QGridLayout(self.scrollAreaWidgetContents)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)

      
        gbox.addWidget(self.image_frame,0, 0, 1, 1)
        gbox.addWidget(self.image_frame2,0, 1, 1, 1)
        gbox.addWidget(self.image_frame3,0, 2, 1, 1)
        gbox.addWidget(self.image_frame4,1, 0, 1, 1)
        gbox.addWidget(self.image_frame5,1, 1, 1, 1)
        gbox.addWidget(self.image_frame0,1, 2, 1, 1)
        gbox.addWidget(self.image_frame6,2, 0, 1, 1)
        gbox.addWidget(self.image_frame7,2, 1, 1, 1)
        gbox.addWidget(self.image_frame8,2, 2, 1, 1)
        gbox.addWidget(self.image_frame9,3, 0, 1, 1)
        gbox.addWidget(self.image_frame10,3, 1, 1, 1)
        

     
        self.layout.addWidget(self.scrollArea)
       
class MouseControlImages(QWidget):

    def __init__(self):
        super().__init__()
        
        self.image_frame = QLabel(self)
        self.image_frame2 = QLabel(self)
        self.image_frame3 = QLabel(self)
        self.image_frame4 = QLabel(self)
        self.image_frame5 = QLabel(self)
        self.image_frame6 = QLabel(self)
        
        color = (0, 0, 255)
        org = (50, 50)
        fontScale = 3
        thickness = 7
        self.image = cv2.imread("Mouse_image\Move.jpeg")
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.image, 'Move', (10,350), font, fontScale, color, thickness, cv2.LINE_AA)
        self.image = cv2.resize(self.image, (200,200))
        self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.image_frame.setPixmap(QtGui.QPixmap.fromImage(self.image))
        
        

        self.image = cv2.imread("Mouse_image\Right.jpeg")
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.image, 'Right', (10,350), font, fontScale, color, thickness, cv2.LINE_AA)
        self.image = cv2.resize(self.image, (200,200))
        self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.image_frame2.setPixmap(QtGui.QPixmap.fromImage(self.image))
       
        

       
        
        self.image = cv2.imread("Mouse_image\Left.jpeg")
        font = cv2.FONT_HERSHEY_SIMPLEX
        self.textLabel = QLabel('Webcam')
        cv2.putText(self.image, 'Left', (10,350), font, fontScale, color, thickness, cv2.LINE_AA)
        self.image = cv2.resize(self.image, (200,200))
        self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.image_frame3.setPixmap(QtGui.QPixmap.fromImage(self.image))
      
        

        self.image = cv2.imread("Mouse_image\Double_click.jpeg")
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.image, 'Double', (10,350), font, fontScale, color, thickness, cv2.LINE_AA)
        self.image = cv2.resize(self.image, (200,200))
        self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.image_frame4.setPixmap(QtGui.QPixmap.fromImage(self.image))
        
        self.image = cv2.imread("Mouse_image\Scroll_up.jpeg")
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.image, 'ScrollUp', (10,350), font, fontScale, color, thickness, cv2.LINE_AA)
        self.image = cv2.resize(self.image, (200,200))
        self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.image_frame5.setPixmap(QtGui.QPixmap.fromImage(self.image))

        self.image = cv2.imread("Mouse_image\Scroll_down.png")
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.image, 'ScrollDwn', (10,450), font, fontScale, color, thickness, cv2.LINE_AA)
        self.image = cv2.resize(self.image, (200,200))
        self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.image_frame6.setPixmap(QtGui.QPixmap.fromImage(self.image))


        gbox = QGridLayout()
        gbox.addWidget(self.image_frame,0, 0, 1, 1)
        gbox.addWidget(self.image_frame2,0, 1, 1, 1)
        gbox.addWidget(self.image_frame3,0, 2, 1, 1)
        gbox.addWidget(self.image_frame4,1, 0, 1, 1)
        gbox.addWidget(self.image_frame5,1, 1, 1, 1)
        gbox.addWidget(self.image_frame6,1, 2, 1, 1)

     
        self.setLayout(gbox)

class AlphabetImages(QWidget):

    def __init__(self):
        super().__init__()
        self.resize(1200,800)
        self.image_frame = QLabel(self)
        self.image_frame2 = QLabel(self)
        self.image_frame3 = QLabel(self)
        self.image_frame4 = QLabel(self)
        self.image_frame5 = QLabel(self)
        self.image_frame6 = QLabel(self)
        self.image_frame7 = QLabel(self)
        self.image_frame8 = QLabel(self)
        self.image_frame9 = QLabel(self)
        self.image_frame10 = QLabel(self)
        self.image_frame11 = QLabel(self)
        self.image_frame12 = QLabel(self)
        self.image_frame13= QLabel(self)
        self.image_frame14 = QLabel(self)
        self.image_frame15 = QLabel(self)
        self.image_frame16 = QLabel(self)
        self.image_frame17 = QLabel(self)
        self.image_frame18 = QLabel(self)
        self.image_frame19 = QLabel(self)
        self.image_frame20 = QLabel(self)
        self.image_frame21 = QLabel(self)
        self.image_frame22 = QLabel(self)
        self.image_frame23= QLabel(self)
        self.image_frame24 = QLabel(self)
        self.image_frame25 = QLabel(self)
        self.image_frame26 = QLabel(self)
        self.image_frame27= QLabel(self)
        self.image_frame28 = QLabel(self)
        self.image_frame29 = QLabel(self)

        color = (0, 0, 255)
        org = (50, 50)
        fontScale = 3
        thickness = 7
        self.image = cv2.imread("AlphabetImages\\5.jpg")
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.image, 'A', (10,350), font, fontScale, color, thickness, cv2.LINE_AA)
        self.image = cv2.resize(self.image, (200,200))
        self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.image_frame.setPixmap(QtGui.QPixmap.fromImage(self.image))
        

        self.image = cv2.imread("AlphabetImages\\10.jpg")
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.image, 'B', (10,350), font, fontScale, color, thickness, cv2.LINE_AA)
        self.image = cv2.resize(self.image, (200,200))
        self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.image_frame2.setPixmap(QtGui.QPixmap.fromImage(self.image))
       
        
        self.image = cv2.imread("AlphabetImages\\6.jpg")
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.image, 'C', (10,350), font, fontScale, color, thickness, cv2.LINE_AA)
        self.image = cv2.resize(self.image, (200,200))
        self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.image_frame3.setPixmap(QtGui.QPixmap.fromImage(self.image))
      
        

        self.image = cv2.imread("AlphabetImages\\13.jpg")
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.image, 'D', (10,350), font, fontScale, color, thickness, cv2.LINE_AA)
        self.image = cv2.resize(self.image, (200,200))
        self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.image_frame4.setPixmap(QtGui.QPixmap.fromImage(self.image))
        

        self.image = cv2.imread("AlphabetImages\\9.jpg")
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.image, 'E', (10,350), font, fontScale, color, thickness, cv2.LINE_AA)
        self.image = cv2.resize(self.image, (200,200))
        self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.image_frame5.setPixmap(QtGui.QPixmap.fromImage(self.image))

        self.image = cv2.imread("AlphabetImages\\3004.jpg")
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.image, 'F', (10,350), font, fontScale, color, thickness, cv2.LINE_AA)
        self.image = cv2.resize(self.image, (200,200))
        self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.image_frame6.setPixmap(QtGui.QPixmap.fromImage(self.image))

        self.image = cv2.imread("AlphabetImages\\30.jpg")
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.image, 'G', (10,350), font, fontScale, color, thickness, cv2.LINE_AA)
        self.image = cv2.resize(self.image, (200,200))
        self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.image_frame7.setPixmap(QtGui.QPixmap.fromImage(self.image))
        
        

        self.image = cv2.imread("AlphabetImages\\28.jpg")
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.image, 'H', (10,350), font, fontScale, color, thickness, cv2.LINE_AA)
        self.image = cv2.resize(self.image, (200,200))
        self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.image_frame8.setPixmap(QtGui.QPixmap.fromImage(self.image))
       
        
        self.image = cv2.imread("AlphabetImages\\20.jpg")
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.image, 'I', (10,350), font, fontScale, color, thickness, cv2.LINE_AA)
        self.image = cv2.resize(self.image, (200,200))
        self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.image_frame9.setPixmap(QtGui.QPixmap.fromImage(self.image))
      
        

        self.image = cv2.imread("AlphabetImages\\34.jpg")
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.image, 'J', (10,350), font, fontScale, color, thickness, cv2.LINE_AA)
        self.image = cv2.resize(self.image, (200,200))
        self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.image_frame10.setPixmap(QtGui.QPixmap.fromImage(self.image))
        

        self.image = cv2.imread("AlphabetImages\\16.jpg")
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.image, 'K', (10,350), font, fontScale, color, thickness, cv2.LINE_AA)
        self.image = cv2.resize(self.image, (200,200))
        self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.image_frame11.setPixmap(QtGui.QPixmap.fromImage(self.image))

        self.image = cv2.imread("AlphabetImages\\14.jpg")
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.image, 'L', (10,350), font, fontScale, color, thickness, cv2.LINE_AA)
        self.image = cv2.resize(self.image, (200,200))
        self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.image_frame12.setPixmap(QtGui.QPixmap.fromImage(self.image))

        self.image = cv2.imread("AlphabetImages\\38.jpg")
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.image, 'M', (10,350), font, fontScale, color, thickness, cv2.LINE_AA)
        self.image = cv2.resize(self.image, (200,200))
        self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.image_frame13.setPixmap(QtGui.QPixmap.fromImage(self.image))
        

        self.image = cv2.imread("AlphabetImages\\105.jpg")
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.image, 'N', (10,350), font, fontScale, color, thickness, cv2.LINE_AA)
        self.image = cv2.resize(self.image, (200,200))
        self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.image_frame14.setPixmap(QtGui.QPixmap.fromImage(self.image))
       
        
        self.image = cv2.imread("AlphabetImages\\21.jpg")
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.image, 'O', (10,350), font, fontScale, color, thickness, cv2.LINE_AA)
        self.image = cv2.resize(self.image, (200,200))
        self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.image_frame15.setPixmap(QtGui.QPixmap.fromImage(self.image))
      
        

        self.image = cv2.imread("AlphabetImages\\35.jpg")
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.image, 'P', (10,350), font, fontScale, color, thickness, cv2.LINE_AA)
        self.image = cv2.resize(self.image, (200,200))
        self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.image_frame16.setPixmap(QtGui.QPixmap.fromImage(self.image))
        

        self.image = cv2.imread("AlphabetImages\\70.jpg")
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.image, 'Q', (10,350), font, fontScale, color, thickness, cv2.LINE_AA)
        self.image = cv2.resize(self.image, (200,200))
        self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.image_frame17.setPixmap(QtGui.QPixmap.fromImage(self.image))

        self.image = cv2.imread("AlphabetImages\\106.jpg")
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.image, 'R', (10,350), font, fontScale, color, thickness, cv2.LINE_AA)
        self.image = cv2.resize(self.image, (200,200))
        self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.image_frame18.setPixmap(QtGui.QPixmap.fromImage(self.image))

        self.image = cv2.imread("AlphabetImages\\37.jpg")
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.image, 'S', (10,350), font, fontScale, color, thickness, cv2.LINE_AA)
        self.image = cv2.resize(self.image, (200,200))
        self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.image_frame19.setPixmap(QtGui.QPixmap.fromImage(self.image))
        
        

        self.image = cv2.imread("AlphabetImages\\47.jpg")
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.image, 'T', (10,350), font, fontScale, color, thickness, cv2.LINE_AA)
        self.image = cv2.resize(self.image, (200,200))
        self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.image_frame20.setPixmap(QtGui.QPixmap.fromImage(self.image))
       
        
        self.image = cv2.imread("AlphabetImages\\303.jpg")
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.image, 'U', (10,350), font, fontScale, color, thickness, cv2.LINE_AA)
        self.image = cv2.resize(self.image, (200,200))
        self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.image_frame21.setPixmap(QtGui.QPixmap.fromImage(self.image))
      
        

        self.image = cv2.imread("AlphabetImages\\110.jpg")
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.image, 'V', (10,350), font, fontScale, color, thickness, cv2.LINE_AA)
        self.image = cv2.resize(self.image, (200,200))
        self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.image_frame22.setPixmap(QtGui.QPixmap.fromImage(self.image))
        

        self.image = cv2.imread("AlphabetImages\\22.jpg")
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.image, 'W', (10,350), font, fontScale, color, thickness, cv2.LINE_AA)
        self.image = cv2.resize(self.image, (200,200))
        self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.image_frame23.setPixmap(QtGui.QPixmap.fromImage(self.image))

        self.image = cv2.imread("AlphabetImages\\140.jpg")
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.image, 'X', (10,350), font, fontScale, color, thickness, cv2.LINE_AA)
        self.image = cv2.resize(self.image, (200,200))
        self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.image_frame24.setPixmap(QtGui.QPixmap.fromImage(self.image))

        
        self.image = cv2.imread("AlphabetImages\\56.jpg")
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.image, 'Y', (10,350), font, fontScale, color, thickness, cv2.LINE_AA)
        self.image = cv2.resize(self.image, (200,200))
        self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.image_frame25.setPixmap(QtGui.QPixmap.fromImage(self.image))
      
        

        self.image = cv2.imread("AlphabetImages\\133.jpg")
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.image, 'Z', (10,350), font, fontScale, color, thickness, cv2.LINE_AA)
        self.image = cv2.resize(self.image, (200,200))
        self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.image_frame26.setPixmap(QtGui.QPixmap.fromImage(self.image))
        

        self.image = cv2.imread("AlphabetImages\\del191.jpg")
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.image, 'del', (10,200), font, fontScale, color, thickness, cv2.LINE_AA)
        self.image = cv2.resize(self.image, (200,200))
        self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.image_frame27.setPixmap(QtGui.QPixmap.fromImage(self.image))

        self.image = cv2.imread("AlphabetImages\\39.jpg")
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.image, 'space', (10,300), font, fontScale, color, thickness, cv2.LINE_AA)
        self.image = cv2.resize(self.image, (200,200))
        self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.image_frame28.setPixmap(QtGui.QPixmap.fromImage(self.image))
        
        self.image = cv2.imread("Numbers_image\Sign10.png")
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.image, 'Close', (10,300), font, fontScale, color, thickness, cv2.LINE_AA)
        self.image = cv2.resize(self.image, (200,200))
        self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.image_frame29.setPixmap(QtGui.QPixmap.fromImage(self.image))

        self.layout = QHBoxLayout(self)
        self.scrollArea= QScrollArea()
        self.widget= QWidget()
        
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QWidget()
        
        
        
        gbox = QGridLayout(self.scrollAreaWidgetContents)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        gbox.addWidget(self.image_frame,0, 0, 1, 1)
        gbox.addWidget(self.image_frame2,0, 1, 1, 1)
        gbox.addWidget(self.image_frame3,0, 2, 1, 1)
        gbox.addWidget(self.image_frame4,0, 3, 1, 1)
        gbox.addWidget(self.image_frame5,0, 4, 1, 1)
        gbox.addWidget(self.image_frame6,1, 0, 1, 1)
        gbox.addWidget(self.image_frame7,1, 1, 1, 1)
        gbox.addWidget(self.image_frame8,1, 2, 1, 1)
        gbox.addWidget(self.image_frame9,1, 3, 1, 1)
        gbox.addWidget(self.image_frame10,1, 4, 1, 1)
        gbox.addWidget(self.image_frame11,2, 0, 1, 1)
        gbox.addWidget(self.image_frame12,2, 1, 1, 1)
        gbox.addWidget(self.image_frame13,2, 2, 1, 1)
        gbox.addWidget(self.image_frame14,2, 3, 1, 1)
        gbox.addWidget(self.image_frame15,2, 4, 1, 1)
        gbox.addWidget(self.image_frame16,3, 0, 1, 1)
        gbox.addWidget(self.image_frame17,3, 1, 1, 1)
        gbox.addWidget(self.image_frame18,3, 2, 1, 1)
        gbox.addWidget(self.image_frame19,3, 3, 1, 1)
        gbox.addWidget(self.image_frame20,3, 4, 1, 1)
        gbox.addWidget(self.image_frame21,4, 0, 1, 1)
        gbox.addWidget(self.image_frame22,4, 1, 1, 1)
        gbox.addWidget(self.image_frame23,4, 2, 1, 1)
        gbox.addWidget(self.image_frame24,4, 3, 1, 1)
        gbox.addWidget(self.image_frame25,4, 4, 1, 1)
        gbox.addWidget(self.image_frame26,5, 0, 1, 1)
        gbox.addWidget(self.image_frame27,5, 1, 1, 1)
        gbox.addWidget(self.image_frame28,5, 2, 1, 1)
        gbox.addWidget(self.image_frame29,5, 3, 1, 1)

     
        self.layout.addWidget(self.scrollArea)

class App(QWidget):
    def __init__(self):
        super(App, self).__init__()
        
        cent = QDesktopWidget().availableGeometry().center()  # Finds the center of the screen
        self.setStyleSheet("background-color: white;")
        self.resize(720, 480)
        self.frameGeometry().moveCenter(cent)
        self.setWindowTitle('S L A I T')
        self.initWindow()
        self.window_1 = MouseControlImages()
        self.window_2 = AlphabetImages()
        self.window_3 = NumbersImage()

    def initWindow(self):
                # create the video capture thread
        self.thread = VideoThread()
        

        self.setWindowTitle("Qt live label demo")
        self.textLabel = QLabel('Webcam')
        self.textLabel.move(30, 550)  # Allocate label in window
        self.textLabel.resize(300, 20)  # Set size for the label
        self.textLabel.setAlignment(Qt.AlignCenter)

        self.button = QPushButton(self)
        self.button.setText('End mouse')   
        
        self.button.clicked.connect(self.Start_mouse)
     
        self.button2 = QPushButton(self)
    
        self.button2.setText('Start Alphabets')
        self.button2.clicked.connect(self.Start_keyboard)
        
        self.button3 = QPushButton(self)
        self.button3.setText('Start Numbers')
        self.button3.clicked.connect(self.Start_Numbers)

        self.button4 = QPushButton(self)
        self.button4.setText('Mouse Control Image')
        self.button4.clicked.connect(self.window1)
        
        self.button5 = QPushButton(self)
        self.button5.setText('Alphabet Gestures Image')
        self.button5.clicked.connect(self.window2)

        self.button6 = QPushButton(self)
        self.button6.setText('Numbers Gestures Image')
        self.button6.clicked.connect(self.window3)
       
        
        
        self.disply_width = 400
        self.display_height = 400
        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)
        self.image_label.move(0, 0)
 
        self.thread.change_pixmap_signal.connect(self.update_image)

        vbox = QVBoxLayout()
        

        hbox = QHBoxLayout()
        hbox.addWidget(self.image_label)  
     

        vbox.addWidget(self.button,5)
        vbox.addWidget(self.button2,5)
        vbox.addWidget(self.button3,5)
        vbox.addWidget(self.button4,5)
        vbox.addWidget(self.button5,5)
        vbox.addWidget(self.button6,5)



        hbox.addLayout(vbox)
        self.setLayout(hbox)
        

        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        # start the thread
        self.thread.start()
   
    def Start_mouse(self):
        global self_thread
        self_thread = 2
        

    def Start_keyboard(self):
        global self_thread
        self_thread = 1
    
    def Start_Numbers(self):
        global self_thread
        self_thread = 3
  
    def window1(self):
        if self.window_1.isVisible():
            self.window_1.hide()
        else:
            self.window_1.show()
    def window2(self):
        if self.window_2.isVisible():
            self.window_2.hide()
        else:
            self.window_2.show()

    def window3(self):
        if self.window_3.isVisible():
            self.window_3.hide()
        else:
            self.window_3.show() 
    def closeEvent(self, event):
       
        self.thread.stop()
        
        event.accept()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)
    
        
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
    
if __name__=="__main__":
    self_thread = 2
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    a = App()
    a.show()
    
    sys.exit(app.exec_())