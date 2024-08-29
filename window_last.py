import sys
import numpy as np
import cv2
from PyQt5 import QtWidgets,QtCore,QtGui
from PyQt5.QtCore import QTimer,QObject,QThread,pyqtSignal
import os
import time   
from threading import Thread
import torch
from ultralytics import YOLO
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['YOLO_VERBOSE'] = 'False'
from ultralytics import YOLO
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")


class DetectionThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    update_text = pyqtSignal(str)

    def __init__(self, capture, model):
        super().__init__()
        self.capture = capture
        self.model = model
        self.is_running = False
        self.pause_detection = False  # 添加一个标志来表示是否暂停检测
        self.detected_labels = []  # 存储检测到的标签
        self.current_labels_str = ''  # 存储当前的标签字符串

    def read_and_process_frame(self):
        ret, frame = self.capture.read()
        if ret:
            results = self.model(frame)[0]
            annotated_frame = results.plot()

            # 清空之前的检测标签
            self.detected_labels.clear()

            # 遍历检测结果
            for detection in results.boxes.data.tolist():
                score, class_id = detection[4], detection[5]  # 置信度和类别 ID
                if score > 0.9:
                    label = self.model.names[int(class_id)]
                    self.detected_labels.append(label)
                    
            # 构造当前的标签字符串
            self.current_labels_str = ', '.join(self.detected_labels)
            self.change_pixmap_signal.emit(annotated_frame)

        else:
            self.stop()

    def run(self):
        while self.is_running:
            self.read_and_process_frame()

    def stop(self):
        self.is_running = False
        self.capture.release()


class MWindow(QtWidgets.QMainWindow):
    update_text = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.UI()
        self.labels_str = ''
        self.timer = QTimer(self)
        self.label_timer = QTimer(self)  # 定时器用于更新标签
        self.label_timer.timeout.connect(self.update_labels)  # 绑定标签更新方法
        self.ss3.clicked.connect(self.toggle_detection)
        self.ss4.clicked.connect(self.stop)
        self.ss1.clicked.connect(self.trans)
        self.ss2.clicked.connect(self.clear)

        self.model = YOLO('lasta.pt')
        self.frame = []
        self.detected_labels = []

        self.capture = None
        self.detection_thread = None

    def toggle_detection(self):
        if self.capture is None:
            self.capture = cv2.VideoCapture(0)  # 使用默认摄像头
        if self.detection_thread is None:
            self.detection_thread = DetectionThread(self.capture, self.model)
            self.detection_thread.change_pixmap_signal.connect(self.update_image)
            self.detection_thread.update_text.connect(self.update_labels)
        if not self.detection_thread.isRunning():
            self.detection_thread.is_running = True
            self.detection_thread.start()
            self.timer.start(30)  # 每30毫秒触发一次，用于更新视频画面
            self.label_timer.start(1000)  # 每1000毫秒（1秒）触发一次，用于更新标签信息
        else:
            self.timer.stop()
            self.label_timer.stop()
            self.detection_thread.is_running = False
            self.detection_thread.stop()

    def stop(self):
        if self.detection_thread is not None:
            self.detection_thread.is_running = False
            self.detection_thread.stop()
            self.label_timer.stop()

    def update_image(self, frame):
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QtGui.QImage(frame.data, width, height, bytes_per_line, QtGui.QImage.Format_BGR888)
        pixmap = QtGui.QPixmap.fromImage(q_image)
        self.label_ori_video.setPixmap(pixmap)

    def update_labels(self):
        if self.detection_thread is not None:
            
        # 获取当前的标签字符串并追加到当前行末尾
            labels_str = self.detection_thread.current_labels_str
            cursor = self.text_edit1.textCursor()
            cursor.movePosition(QtGui.QTextCursor.End)  # 将光标移动到文本框末尾
            self.text_edit1.setTextCursor(cursor)
            if labels_str == "空格":
                labels_str = " "    
            elif labels_str == "句号":
                labels_str = "."
            # 删除光标前的一个字符
            cursor.deletePreviousChar()
            self.text_edit1.setTextCursor(cursor)
            self.text_edit1.setFontPointSize(18)
            self.text_edit1.insertPlainText(labels_str + "_")  # 插入标签字符串，并在末尾添加空格
            self.text_edit1.setTextCursor(cursor)


    def trans(self):
        # 分词

        source_language_code = "eng_Latn"  # 英语
        target_language_code = "zho_Hans"  # 中文

        # Input text to translate
        text = self.text_edit1.toPlainText()
        self.text_edit2.append(text)

        # Prepend the source language prefix
        inputs = tokenizer(f"<{source_language_code}> {text}", return_tensors="pt")

        # Generate the translation
        translated_tokens = model.generate(inputs["input_ids"], forced_bos_token_id=tokenizer.lang_code_to_id[target_language_code])

        # Decode the translated tokens
        translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

        self.text_edit2.clear()
        self.text_edit2.setFontPointSize(18)
        self.text_edit2.append(str(translated_text))
    def clear(self):
        cursor = self.text_edit1.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)  # 将光标移动到文本框末尾
        
        # 删除光标前的一个字符
        cursor.deletePreviousChar()
        
        # 更新光标位置
        self.text_edit1.setTextCursor(cursor)


    def UI(self):
        self.setStyleSheet(""" 
            QWidget {
                background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 rgba(255, 192, 203, 255), stop:1 rgba(255, 105, 180, 255));
                border-radius: 10px;
            }
        """)
        self.resize(640 , 800)

        # central Widget
        centralWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(centralWidget)

        mainLayout = QtWidgets.QVBoxLayout(centralWidget)

        # 子 layout
        topLayout = QtWidgets.QHBoxLayout()
        self.label_ori_video=QtWidgets.QLabel(self)

        self.label_ori_video.setMinimumSize(640,360)

        # self.label_ori_video.setStyleSheet('border:1px solid')
        self.label_ori_video.setStyleSheet("border-radius: 10px;")

        topLayout.addWidget(self.label_ori_video)


        bottomLayout = QtWidgets.QHBoxLayout()

        mainLayout.addLayout(topLayout)
        mainLayout.addLayout(bottomLayout)# 添加 子 Layout

        btnLayout=QtWidgets.QVBoxLayout()
        self.ss1 = CustomButton('😂😂翻译')
        self.ss2 = CustomButton('😂😂清除')
        self.ss3 = CustomButton('😂😂摄像')
        self.ss4 = CustomButton('😂😂停止')
        btnLayout.addWidget(self.ss1)
        btnLayout.addWidget(self.ss2)
        btnLayout.addWidget(self.ss3)
        btnLayout.addWidget(self.ss4)
        bottomLayout.addLayout(btnLayout)

        self.text_edit1 = QtWidgets.QTextEdit(self)
        self.text_edit1.setReadOnly(True)
        self.text_edit2 = QtWidgets.QTextEdit(self)
        self.text_edit2.setReadOnly(True)
        mainLayout.addWidget(self.text_edit1)
        mainLayout.addWidget(self.text_edit2)


class CustomButton(QtWidgets.QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setStyleSheet("""
            QPushButton {
                background-color: #FFC0CB;
                border-radius: 10px;
                padding: 5px;
                font-size: 14px;
                color: #000000;
            }
            QPushButton:hover {
                background-color: #FF69B4;
            }
            QPushButton:pressed {
                background-color: #FF1493;
            }
        """)

    
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MWindow()
    window.show()
    sys.exit(app.exec_())