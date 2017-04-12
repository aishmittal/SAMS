import os
import sys
import csv
import cv2
import time
from PyQt4 import QtGui, QtCore
from PyQt4.QtGui import *
from PyQt4.QtCore import *
import sqlite3

import database_operations as dsop
import cnn_model_generator as cmg




# Global Variables
base_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(base_path,'data')
dataset_path = os.path.join(data_path,'datasets')
image_path = os.path.join(data_path,'images')
model_path = os.path.join(data_path,'models')
tmp_path = os.path.join(base_path,'tmp')
camera_port = 1
large_text_size = 28
medium_text_size = 18
small_text_size = 10
demoVar = "Physics Class"
window_width = 600
window_height = 250
conn = sqlite3.connect('sams.db')
cursor = conn.cursor()
new_user_added = False
current_subject_code = ''
current_student_enroll_no = 0
current_model_path = ''
current_model_md_path = ''


def query(comm,params=()):
    cursor.execute(comm,params)
    conn.commit()
    return cursor

def multiple_select_query(comm,params=()):
    cursor.execute(comm,params)
    res = cursor.fetchall()
    return res


def select_query(comm,params=()):
    cursor.execute(comm,params)
    res = cursor.fetchall()
    res =[x[0] for x in res]
    return res




class WindowMain(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.setStyleSheet("#gframe {border-radius:5px;border:1px solid #a5a5a5}")

    def initUI(self):
        self.setWindowTitle("SAMs - Smart Attendance Management system")
        self.setFixedSize(window_width, window_height)
  
        self.move(QApplication.desktop().screen().rect().center()- self.rect().center())
        self.vbox = QVBoxLayout()
        # self.vbox.setAlignment(Qt.AlignCenter)

        self.button1 = QPushButton("Register")
        self.button1.clicked.connect(self.onclick_button1)

        self.button2 = QPushButton("Take Attendance")
        self.button2.clicked.connect(self.onclick_button2)

        self.button3 = QPushButton("Show Attendance Records")
        self.button3.clicked.connect(self.onclick_button3)

        self.button4 = QPushButton("Exit")
        self.button4.clicked.connect(self.quitApp)

        self.button1.setFixedWidth(400)
        self.button2.setFixedWidth(400)
        self.button3.setFixedWidth(400)
        self.button4.setFixedWidth(400)

        self.vbox.addWidget(self.button1)
        self.vbox.addWidget(self.button2)
        self.vbox.addWidget(self.button3)
        self.vbox.addWidget(self.button4)
        self.vbox.setContentsMargins(100, 30, 100, 30)
        self.vbox.setSpacing(15)

        self.frame = QFrame()
        
        self.frame.setObjectName('gframe')
        self.frame.setLayout(self.vbox)

        self.hbox = QHBoxLayout()
        self.hbox.setAlignment(Qt.AlignCenter)
        self.hbox.addWidget(self.frame)

        # self.addWidget(self.frame)
        self.setLayout(self.hbox)

    def setGeom(self,geom):
        self.setGeometry(geom)

    def onclick_button1(self):
        # Register the student
        winRegister.setGeometry(winRegister.frameGeometry())
        win.hide()
        winRegister.show()

    def onclick_button2(self):
        # Take Attendance
        winTakeAttnSelectClass.setGeometry(winTakeAttnSelectClass.frameGeometry())
        win.hide()
        winTakeAttnSelectClass.reset()
        winTakeAttnSelectClass.show()

    def onclick_button3(self):
        # Take Attendance
        winShowAttnSelectClass.setGeometry(winShowAttnSelectClass.frameGeometry())
        win.hide()
        winShowAttnSelectClass.reset()
        winShowAttnSelectClass.show()

    def quitApp(self):
        quit_msg = "Are you sure you want to exit the program?"
        reply = QMessageBox.question(self, 'Message', 
                         quit_msg, QMessageBox.Yes, QMessageBox.No)

        if reply == QMessageBox.Yes:
            QtCore.QCoreApplication.instance().quit()




class WindowRegister(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        

    def initUI(self):
        self.setWindowTitle("Register Subject")
        self.resize(window_width, window_height)
        self.move(QApplication.desktop().screen().rect().center()- self.rect().center())
        self.setStyleSheet("#gframe {border-radius:5px;border:1px solid #a5a5a5}")

        self.fbox = QFormLayout()
        self.vbox = QVBoxLayout()

        self.textLabel1 = QLabel("Subject Name: ", self)
        self.textLabel2 = QLabel("Subject Code: ", self)
        self.textLabel3 = QLabel("No. of Students: ", self)
        self.messageLbl = QLabel('')
        font1 = QFont('Helvetica', small_text_size)
        self.messageLbl.setFont(font1)
        
        self.lineEdit1 = QLineEdit(self)
        self.lineEdit2 = QLineEdit(self)
        self.lineEdit3 = QLineEdit(self)

        self.hbox = QHBoxLayout()
        
        self.nextButton = QPushButton("Next", self)
        self.nextButton.clicked.connect(self.next)

        self.backButton = QPushButton("Back", self)
        self.backButton.clicked.connect(self.back)

        self.hbox.addWidget(self.nextButton)
        self.hbox.addWidget(self.backButton)
        self.hbox.setObjectName('gframe')

        self.fbox.addRow(self.textLabel1,self.lineEdit1)
        self.fbox.addRow(self.textLabel2,self.lineEdit2)
        self.fbox.addRow(self.textLabel3,self.lineEdit3)
        self.fbox.addRow(self.messageLbl)
        
        self.fbox.setContentsMargins(80, 40, 80, 40)
        self.fbox.setSpacing(10)
        self.form = QFrame()
        self.form.setLayout(self.fbox)
        self.form.setObjectName('gframe')

        self.buttons = QFrame()
        self.buttons.setLayout(self.hbox)
        # self.buttons.setObjectName('gframe')

        self.fbox.setAlignment(Qt.AlignCenter)
        self.vbox.setAlignment(Qt.AlignCenter)
        
        self.vbox.addWidget(self.form)
        self.vbox.addWidget(self.buttons)

        self.setLayout(self.vbox)

    def back(self):
        self.lineEdit1.clear()
        self.lineEdit2.clear()
        self.lineEdit3.clear()
        win.setGeometry(win.frameGeometry())
        win.setGeometry(win.frameGeometry())
        winRegister.hide()
        # time.sleep(0.05)

        win.show()

    def next(self):
        # Start the Registration process
        if (not self.lineEdit1.text()) or (not self.lineEdit2.text()) or (not self.lineEdit3.text()):
            self.messageLbl.setText('Error: One or more required fields empty! verification failed')
            return

        elif int(self.lineEdit3.text())<=1:
            self.messageLbl.setText('Error: Require more than 1 student for a course!')
            return

        else:
            subject_name = str(self.lineEdit1.text())
            subject_code = str(self.lineEdit2.text())
            no_of_students = str(self.lineEdit3.text())
            # print(subject_name,subject_code,no_of_students)
            
            sql_command = """SELECT * FROM subjects WHERE subject_code = ? """
            cursor.execute(sql_command,(subject_code,))
            
            if cursor.fetchone():
                self.messageLbl.setText('Error: Subject already exist!')
                return
            
            else:
                self.messageLbl.setText('Success: Verification Successful!')
                
                format_str = """INSERT INTO subjects (subject_id,subject_name,subject_code,no_of_students) 
                     VALUES (NULL,?,?,?);"""
                params = (subject_name, subject_code,no_of_students)         
                conn.execute(format_str,params)
                conn.commit()
                
                global current_subject_code
                current_subject_code = subject_code
                self.messageLbl.setText('Success: Subject added to database sucessfully!')
                subj_dir = os.path.join(image_path,subject_code)
                
                if not os.path.exists(subj_dir):
                    os.makedirs(subj_dir)
                # print(int(no_of_students))
                winRegisterStudentPhotos.newSubjectReg(int(no_of_students))
                winRegisterStudentDetails.newSubjectReg(int(no_of_students))
            
                self.lineEdit1.clear()
                self.lineEdit2.clear()
                self.lineEdit3.clear()
                winRegisterStudentDetails.setGeometry(winRegisterStudentDetails.frameGeometry())
                winRegister.hide()
                winRegisterStudentDetails.setWindowTitle('Add details of student 1 (Remaining %d)'%(int(no_of_students)-1))
                winRegisterStudentDetails.show()



class WindowRegisterStudentDetails(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
        self.reg_count = 0
        self.total_student_count = 0

    def initUI(self):
        self.setWindowTitle("Enter Student Details")
        self.resize(window_width, window_height)
        self.move(QApplication.desktop().screen().rect().center()- self.rect().center())
        self.fbox = QFormLayout()

        self.textLabel1 = QLabel("Enrollment No: ", self)
        self.textLabel2 = QLabel("Student Name: ", self)
        self.messageLbl = QLabel('')
        self.lineEdit1 = QLineEdit(self)
        self.lineEdit2 = QLineEdit(self)
        self.button1 = QPushButton("Next", self)
        self.button1.clicked.connect(self.onclick_button1)

        self.fbox.addRow(self.textLabel1,self.lineEdit1)
        self.fbox.addRow(self.textLabel2,self.lineEdit2)
        self.fbox.addRow(self.messageLbl)
        self.fbox.addRow(self.button1)
        self.fbox.setAlignment(Qt.AlignCenter)
        self.fbox.setContentsMargins(100, 50, 100, 50)
        self.fbox.setSpacing(10)
        self.setLayout(self.fbox)

    def onclick_button1(self):
        if ((not self.lineEdit1.text()) or (not self.lineEdit2.text())) and current_subject_code:
            self.messageLbl.setText('Error: One or more required fields empty! verification failed')
            return
        else:
            enroll_no = str(self.lineEdit1.text())
            student_name = str(self.lineEdit2.text())
            print("Added: ", enroll_no, student_name)
            sql_command = """SELECT * FROM reg_students WHERE enroll_no = ? AND subject_code = ? """
            cursor.execute(sql_command,(enroll_no,current_subject_code))
            if cursor.fetchone():
                self.messageLbl.setText('Error: Student already registered for this subject!')
                return
            else:
                self.messageLbl.setText('Success: Verification Successful!')
                format_str = """INSERT INTO reg_students (student_id,enroll_no,student_name,subject_code) 
                     VALUES (NULL,?,?,?);"""
                params = (enroll_no,student_name,current_subject_code)         
                conn.execute(format_str,params)
                conn.commit()
                global current_student_enroll_no
                current_student_enroll_no = enroll_no
                self.messageLbl.setText('Success: Student registered for subject sucessfully!')
                self.store_dir = os.path.join(image_path,current_subject_code,str(current_student_enroll_no))
                if not os.path.exists(self.store_dir):
                    os.makedirs(self.store_dir)

                
        # Forward to next page
        self.messageLbl.clear()
        self.lineEdit1.clear()
        self.lineEdit2.clear()
        winRegisterStudentPhotos.setGeometry(winRegisterStudentPhotos.frameGeometry())
        winRegisterStudentDetails.hide()
        winRegisterStudentPhotos.setWindowTitle('Add Face Images of student %d (Remaining %d)'%(self.reg_count+1,self.total_student_count-self.reg_count-1))
        winRegisterStudentPhotos.show()


    def newSubjectReg(self,n):
        self.reg_count = 0
        self.total_student_count = n

    def newStudentAdded(self):
        self.reg_count = self.reg_count+1


class WindowRegisterStudentPhotos(QWidget):
    def __init__(self):
        super().__init__()
        self.capturing=False
        self.setFixedSize(700,460)
        self.move(QApplication.desktop().screen().rect().center()- self.rect().center())
        self.video_size = QSize(400, 300)
        self.snapshot_size = QSize(80, 80)
        self.cascPath = 'haarcascade_frontalface_default.xml'
        self.faceCascade = cv2.CascadeClassifier(self.cascPath)
        self.snapshotCnt=0
        self.maxSnapshotCnt=8
        self.captureCompleted = False
        self.reg_count = 0
        self.total_student_count = 0
        self.setStyleSheet("#gframe {border-radius:5px;border:1px solid #a5a5a5}")
        self.initUI()

    def initUI(self):
        self.topleft = QFrame()        
        self.imageLabel=QLabel()
        self.imageLabel.setScaledContents(True)
        self.topleft.setObjectName('gframe')
        self.topleft.setContentsMargins(50,10,50,10)
        self.imageLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.vbox1 = QVBoxLayout()
        self.vbox1.addWidget(self.imageLabel)
        self.topleft.setLayout(self.vbox1)

        self.topright = QFrame()
        self.snpGrid = QGridLayout()
        
        self.snpGrid.setSpacing(2)
        self.snpGrid.setContentsMargins(2,2,2,2)
        
        self.topright.setLayout(self.snpGrid)
        self.hbox = QHBoxLayout()
        self.startButton = QPushButton('Start')
        self.stopButton = QPushButton('Stop')
        self.takeSnapshotButton = QPushButton('Take Snapshot')
        self.nextButton = QPushButton('Next')
        self.trainModelButton = QPushButton('Train Model')
        self.messageLbl = QLabel('')
        font1 = QFont('Helvetica', small_text_size)
        self.messageLbl.setFont(font1)

        self.startButton.clicked.connect(self.startCapture)
        self.stopButton.clicked.connect(self.stopCapture)
        self.takeSnapshotButton.clicked.connect(self.takeSnapshot)
        self.nextButton.clicked.connect(self.nextStudent)
        self.trainModelButton.clicked.connect(self.trainModel)
        self.hbox.addWidget(self.startButton)
        self.hbox.addWidget(self.stopButton)
        self.hbox.addWidget(self.takeSnapshotButton)
        self.hbox.addWidget(self.nextButton)
        # self.hbox.addWidget(self.trainModelButton)
        
        self.mhbox = QHBoxLayout()
        self.mhbox.setAlignment(Qt.AlignCenter)
        self.mhbox.addWidget(self.messageLbl)

        self.bvbox = QVBoxLayout()
        self.bvbox.addLayout(self.mhbox)
        self.bvbox.addLayout(self.hbox)
        self.bvbox.setSpacing(10)
        
        self.bottom = QFrame()
        self.bottom.setLayout(self.bvbox)
        self.bottom.setObjectName("gframe")

        self.splitter1 = QSplitter(Qt.Horizontal)
        self.splitter1.addWidget(self.topleft)
        self.splitter1.addWidget(self.topright)
        self.splitter1.setSizes([5,2])

        self.splitter2 = QSplitter(Qt.Vertical)
        self.splitter2.addWidget(self.splitter1)
        self.splitter2.addWidget(self.bottom)
        self.splitter2.setSizes([375,75])
        self.hbox1=QHBoxLayout()
        self.hbox1.addWidget(self.splitter2)
        self.setLayout(self.hbox1)
        self.initGrid()

    def initDir(self):
        self.store_dir = os.path.join(image_path,current_subject_code,str(current_student_enroll_no))
        if os.path.isdir(self.store_dir)==False:
            try:
                original_umask = os.umask(0)
                os.makedirs(self.store_dir)
            finally:
                os.umask(original_umask)

    def initGrid(self):
        range_x=int((self.maxSnapshotCnt+1)/2)
        self.snpLabels =[]
        for i in range(self.maxSnapshotCnt):
            self.snpLabels.append(QLabel())
            self.snpLabels[i].setScaledContents(True)
            self.snpLabels[i].setFixedSize(self.snapshot_size)
            self.snpLabels[i].setObjectName("gframe")

        range_y =2
        pos = [(i,j) for i in range(range_x) for j in range(range_y)]
        
        for p, lbl in zip(pos, self.snpLabels):
            self.snpGrid.addWidget(lbl,*p)


    def display_video_stream(self):
        r , frame = self.capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40)
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)
        image = QImage(frame, frame.shape[1], frame.shape[0], 
                       frame.strides[0], QImage.Format_RGB888)
        
        self.imageLabel.setPixmap(QPixmap.fromImage(image))

    def nextStudent(self):

        self.messageLbl.clear()
        for i in range(self.maxSnapshotCnt):
            self.snpLabels[i].clear()
        self.snapshotCnt = 0
        self.captureCompleted = False  
        self.capturing = False  
        self.imageLabel.clear()
        self.reg_count = self.reg_count + 1
        if self.reg_count == self.total_student_count-1:
            self.nextButton.setText('Train Model')

        if self.reg_count < self.total_student_count:
            winRegisterStudentDetails.setGeometry(winRegisterStudentDetails.frameGeometry())
            winRegisterStudentPhotos.hide()
            winRegisterStudentDetails.setWindowTitle('Add details of student %d (Remaining %d)'%(self.reg_count+1,self.total_student_count-self.reg_count-1))
            winRegisterStudentDetails.show()

        else:
            self.trainModel()    

    def newSubjectReg(self,n):
        self.reg_count = 0
        self.total_student_count = n


    def trainModel(self):
     
        LOAD_PATH= os.path.join(image_path,current_subject_code)
        DSETP=dataset_path
        MODLP=model_path

        DSETFN=current_subject_code
        MFN="model_"+DSETFN# model filename

        NEPOCH=10
        DBS=500# data batch size
        TBS=10# training batch size
        DMY="no val"# dummmy
        try:

            print('-------------------- Model Training Started ----------------------')
            dsop.createSingleBlockDataset(LOAD_PATH,DSETP,DSETFN,(50,50,3))
            md=dsop.loadMetaData(DSETP+'/'+DSETFN+'_metadata.txt')
            print(md[0],md[0]["shape"])

            print("PATH:",DSETP+"/"+DSETFN+".h5",md[0]["shape"])

            dsop.partitionDataset(DSETP+"/"+DSETFN+".h5",DSETP+"/"+DSETFN+"_metadata.txt",(80,20))

            md=dsop.loadMetaData(DSETP+'/'+DSETFN+'_train_metadata.txt')
            model= cmg.getModelFrame(md[0]["shape"],int(md[0]["nb_classes"]),3)
            DBS= md[0]["dataset_shape"][0]
            MFN=MFN+"_"+str(NEPOCH)


            MODEL_LOC=MODLP+"/"+MFN+".h5"
            TD_LOC=DSETP+"/"+DSETFN+"_test.h5"
            TD_MD_LOC=DSETP+"/"+DSETFN+"_test_metadata.txt"

            trained_model_path, model_md_path=cmg.getCustomOptimalTrainedModel(model,DSETP+"/"+DSETFN+"_train.h5",
                                                                       DSETP+"/"+DSETFN+"_train_metadata.txt",
                                                                 MODLP,MFN,70,2,
                                                                 0.8,15,0.2,TD_LOC,TD_MD_LOC,1000)
            print(trained_model_path)
            model_name = os.path.basename(trained_model_path)
            format_str = """UPDATE subjects SET model_name = ? WHERE subject_code = ?"""             
            params = (model_name, current_subject_code)         
            conn.execute(format_str,params)
            conn.commit()


            MODEL_LOC=MODLP+"/"+MFN+".h5"
            TD_LOC=DSETP+"/"+DSETFN+"_test.h5"
            TD_MD_LOC=DSETP+"/"+DSETFN+"_test_metadata.txt"

            cmg.evaluateModel (trained_model_path,TD_LOC,TD_MD_LOC)
            print('-------------------- Model Training Completed --------------------')

        except Exception as e:
                self.messageLbl.setText('Error: Some error while trainig model! check console')
                print("Model Training Failed...\n Errors:")
                print(e)

        self.finishRegistration()


    def finishRegistration(self):
        try:
            sql_command = "SELECT enroll_no FROM reg_students WHERE subject_code = '%s' ORDER BY enroll_no" % (current_subject_code)        
            enroll_list = select_query(sql_command)
            col = ""
            for i in enroll_list:
                col+=",\n'%s' INTEGER NOT NULL DEFAULT 0" % (str(i))
            
            sql_command = """CREATE TABLE IF NOT EXISTS %s (
                        id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                        date_and_time DATETIME NOT NULL DEFAULT (datetime('now','localtime'))%s
                        );""" %(current_subject_code,col)
            query(sql_command)
            print('-------------------- Registration Finished Successfully --------------------')

        except Exception as e:
                print("Error while creating attendance table...\n Errors:")
                print(e)

        winRegisterStudentPhotos.setGeometry(winRegisterStudentPhotos.frameGeometry())
        winRegisterStudentPhotos.hide()
        win.show()


    def startCapture(self):
        self.initDir()
        self.capturing = True
        self.capture = cv2.VideoCapture(camera_port)
        #self.capture.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, self.video_size.width())
        #self.capture.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, self.video_size.height())
        
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.display_video_stream)
        self.timer.start(30)

    def stopCapture(self):
        #print "pressed End"
        if self.capturing == True:
            self.capturing = False
            self.capture.release()
            self.timer.stop()
            cv2.destroyAllWindows()

    def takeSnapshot(self):

        if self.capturing == False:
            self.messageLbl.setText('Warning: Start the camera')
            return

        if self.snapshotCnt == self.maxSnapshotCnt:
            self.messageLbl.setText('Warning: All snapshots taken, no need to take more now!')
            return                 
        
        if (self.capturing == True)  and (self.snapshotCnt < self.maxSnapshotCnt):
            try:
                r , frame = self.capture.read()
                frame = cv2.flip(frame, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.faceCascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(40, 40)
                )
                if len(faces)==0:
                    return
                max_area = 0
                mx = 0
                my = 0 
                mh = 0 
                mw = 0
                for (x, y, w, h) in faces:
                    if w*h > max_area:
                        mx = x
                        my = y
                        mh = h
                        mw = w
                        max_area=w*h    
                
                image_crop = frame[my:my+mh,mx:mx+mw]
                self.snapshotCnt=self.snapshotCnt+1
                self.messageLbl.setText('Process: Total snapshots captured: %d (Remaining: %d)' % (self.snapshotCnt,self.maxSnapshotCnt-self.snapshotCnt))
                file_name = 'img_%d.jpg'% (self.snapshotCnt)
                file = os.path.join(self.store_dir,file_name)
                cv2.imwrite(file, image_crop)
                self.snpLabels[self.snapshotCnt-1].setPixmap(QPixmap(file))

            except Exception as e:
                self.messageLbl.setText('Error: Snapshot capturing failed')
                print("Snapshot capturing failed...\n Errors:")
                print(e)

        if(self.snapshotCnt == self.maxSnapshotCnt):
            self.captureCompleted=True
            self.stopCapture()




class WindowTakeAttnSelectClass(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        

    def initUI(self):
        self.setWindowTitle("Take Attendance")
        self.resize(window_width, window_height+100)
        self.move(QApplication.desktop().screen().rect().center()- self.rect().center())
        self.vbox = QVBoxLayout()
        self.subjectSelectLbl = QLabel('Select Subject')
   
        self.subjectSelect = QListWidget()
        # self.subjectSelect.setSortingEnabled(True)

        self.hbox = QHBoxLayout()

        self.startButton = QPushButton("Start", self)
        self.startButton.clicked.connect(self.startClass)
        self.backButton = QPushButton("Back", self)
        self.backButton.clicked.connect(self.back)
        
        self.vbox2 = QVBoxLayout()
        self.vbox2.setAlignment(Qt.AlignCenter)
        self.modeLbl = QLabel('Select Attendance Mode')
        self.automaticButton = QRadioButton('Automatic')
        self.automaticButton.setChecked(True)
        self.manualButton = QRadioButton('Manual')
        self.delayLbl = QLabel('Time Delay (in seconds)')
        self.timeDelay = QSpinBox()
        self.timeDelay.setRange(10,120)
        self.timeDelay.setSingleStep(10)
     

        self.hbox2=QHBoxLayout()
        self.hbox3=QHBoxLayout()

        self.hbox2.addWidget(self.automaticButton)
        self.hbox2.addWidget(self.manualButton)
        self.hbox3.addWidget(self.delayLbl)
        self.hbox3.addWidget(self.timeDelay)
        self.hbox3.addStretch()

        self.vbox2.addWidget(self.modeLbl)
        
        self.vbox2.addLayout(self.hbox2)
        self.vbox2.addLayout(self.hbox3)

        self.hbox.addWidget(self.startButton)
        self.hbox.addWidget(self.backButton)
        
        self.vbox.addWidget(self.subjectSelectLbl)
        self.vbox.addWidget(self.subjectSelect)
        self.vbox.addLayout(self.vbox2)
        self.vbox.addLayout(self.hbox)
        self.vbox.setAlignment(Qt.AlignCenter)
        self.setLayout(self.vbox)
        self.reset()

    def reset(self):
        self.subjectSelect.clear()
        sql_command = """SELECT * FROM subjects"""
        self.res = multiple_select_query(sql_command)
        if len(self.res)>0:
            for i in range(0,len(self.res)):
                self.subjectSelect.addItem(self.res[i][1] + ' (' + self.res[i][2]+ ') ')
        self.automaticButton.setChecked(True)
        self.timeDelay.setValue(60)
        self.subjectSelect.setCurrentRow(0)


        
    def back(self):
        win.setGeometry(win.frameGeometry())
        winTakeAttnSelectClass.hide()
        win.show()
                
    def startClass(self):
        # Check which Class is selected!
        global current_subject_code
        global current_model_path
        global current_model_md_path
        # print(self.subjectSelect.currentRow())
        subject_name = self.subjectSelect.currentItem().text()
        subject_code = self.res[self.subjectSelect.currentRow()][2]
        print("Selected Subject: ",subject_name)

        # print(subject_name,subject_code)
        
        current_subject_code = subject_code
        sql_command = "SELECT model_name FROM subjects WHERE subject_code = '%s' " % (current_subject_code)
        cursor.execute(sql_command)
        self.res2 = cursor.fetchone()
        model_name  = self.res2[0]
        # print(model_name) 
        
        current_model_path = os.path.join(model_path,model_name)
        current_model_md_path = os.path.join(model_path,model_name.split('.')[0]+'_metadata.txt')
        print("Model Path:",current_model_path)
        print("Model Metadata Path:",current_model_md_path)
       
        
        # print(sql_command)            
        try:
            sql_command = "SELECT enroll_no FROM reg_students WHERE subject_code = '%s' ORDER BY enroll_no" % (current_subject_code)        
            enroll_list = select_query(sql_command)
            col = ""
            for i in enroll_list:
                col+=",\n'%s' INTEGER NOT NULL DEFAULT 0" % (str(i))
            
            sql_command = """CREATE TABLE IF NOT EXISTS %s (
                        id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                        date_and_time DATETIME NOT NULL DEFAULT (datetime('now','localtime'))%s
                        );""" %(current_subject_code,col)
            query(sql_command)

        except Exception as e:
                print("Error while creating attendance table...\n Errors:")
                print(e)

        winTakeAttnGroupPhotos.setGeometry(winTakeAttnGroupPhotos.frameGeometry())
        winTakeAttnSelectClass.hide()
        
        if self.automaticButton.isChecked() == True:
            winTakeAttnGroupPhotos.setMode(0,self.timeDelay.value())
        else:
            winTakeAttnGroupPhotos.setMode(1,self.timeDelay.value())

        self.reset()     

        
        winTakeAttnGroupPhotos.show()
        winTakeAttnGroupPhotos.setWindowTitle("Take Attendance Photos for " + subject_name + " class")
        



class WindowTakeAttnGroupPhotos(QWidget):
    def __init__(self):
        super().__init__()
        self.capturing=False
        self.setFixedSize(700,460)
        self.move(QApplication.desktop().screen().rect().center()- self.rect().center())
        self.video_size = QSize(400, 300)
        self.snapshot_size = QSize(80, 80)
        self.cascPath = 'haarcascade_frontalface_default.xml'
        self.faceCascade = cv2.CascadeClassifier(self.cascPath)
        self.attendance_record={}
        self.setStyleSheet("#gframe {border-radius:5px;border:1px solid #a5a5a5}")
        self.initUI()

    def initUI(self):
        self.top = QFrame()        
        self.imageLabel=QLabel()
        self.imageLabel.setScaledContents(True)
        self.top.setObjectName('gframe')
        self.top.setContentsMargins(50,10,50,10)
        self.imageLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.vbox1 = QVBoxLayout()
        self.vbox1.addWidget(self.imageLabel)
        self.top.setLayout(self.vbox1)

        self.hbox = QHBoxLayout()
        self.startButton = QPushButton('Start')
        self.stopButton = QPushButton('Stop')
        self.takeSnapshotButton = QPushButton('Take Snapshot')
        self.endClassButton = QPushButton('End Class')
        self.messageLbl = QLabel('')
        font1 = QFont('Helvetica', small_text_size)
        self.messageLbl.setFont(font1)

        self.startButton.clicked.connect(self.startCapture)
        self.stopButton.clicked.connect(self.stopCapture)
        self.takeSnapshotButton.clicked.connect(self.takeSnapshot)
        self.endClassButton.clicked.connect(self.endClass)
        self.hbox.addWidget(self.startButton)
        self.hbox.addWidget(self.stopButton)
        self.hbox.addWidget(self.takeSnapshotButton)
        self.hbox.addWidget(self.endClassButton)
        
        self.mhbox = QHBoxLayout()
        self.mhbox.setAlignment(Qt.AlignCenter)
        self.mhbox.addWidget(self.messageLbl)

        self.bvbox = QVBoxLayout()
        self.bvbox.addLayout(self.mhbox)
        self.bvbox.addLayout(self.hbox)
        self.bvbox.setSpacing(10)
        
        self.bottom = QFrame()
        self.bottom.setLayout(self.bvbox)
        self.bottom.setObjectName("gframe")

        self.splitter2 = QSplitter(Qt.Vertical)
        self.splitter2.addWidget(self.top)
        self.splitter2.addWidget(self.bottom)
        self.splitter2.setSizes([375,75])
        self.hbox1=QHBoxLayout()
        self.hbox1.addWidget(self.splitter2)
        self.setLayout(self.hbox1)
        self.reset()

    def reset(self):
        self.imageLabel.clear()
        self.time_delay=60
        self.mode = 0
        self.snapshotCnt=0
        self.attendance_record = {}
     


    def setMode(self,mode,time_delay=1):
        print('Mode:' ,mode)
        print('Time Delay: ',time_delay)
        self.mode = mode
        self.time_delay = time_delay
        if self.mode == 0:
            self.takeSnapshotButton.setEnabled(False)
        else:
            self.takeSnapshotButton.setEnabled(True)



    def display_video_stream(self):
        r , frame = self.capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40)
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)
        image = QImage(frame, frame.shape[1], frame.shape[0], 
                       frame.strides[0], QImage.Format_RGB888)
        
        self.imageLabel.setPixmap(QPixmap.fromImage(image))
  

    def startCapture(self):
        if not self.capturing: 
            self.capturing = True
            self.snapshotCnt=0
            print("--------------- Class Started -----------------")
            if len(self.attendance_record)==0:
                self.attendance_record = {}
                sql_command = "SELECT enroll_no FROM reg_students WHERE subject_code = '%s' ORDER BY enroll_no" % (current_subject_code)
                # cursor.execute(sql_command)
                # res = cursor.fetchall()
                res = select_query(sql_command)
                print(res)
                if len(res)>0:
                    # print(res)
                    for i in res:
                        self.attendance_record[str(i)]=0

                # print(self.attendance_record)
            self.capture = cv2.VideoCapture(camera_port)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.display_video_stream)
        self.timer.start(30)
        if self.mode == 0:
            self.automaticCapture()

    def automaticCapture(self):
        self.timer2 = QtCore.QTimer()
        self.timer2.timeout.connect(self.takeSnapshot)
        self.timer2.start(self.time_delay*1000)

    def stopCapture(self):
        #print "pressed End"
        if self.capturing == True:
            self.capturing = False
            self.capture.release()
            self.timer.stop()
            if self.mode == 0:
                self.timer2.stop()                
            cv2.destroyAllWindows()

    def takeSnapshot(self):

        if self.capturing == False:
            self.messageLbl.setText('Warning: Start the camera')
            return
              
        
        if (self.capturing == True):
            try:
                r , frame = self.capture.read()
                frame = cv2.flip(frame, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.faceCascade.detectMultiScale(
                    gray,
                    scaleFactor=1.3,
                    minNeighbors=5
                )
                print("\n--------------- Image Taken -----------------")

                if len(faces)>0:

                    self.snapshotCnt=self.snapshotCnt+1
                    pred_results=cmg.labelFaces(current_model_path,current_model_md_path,frame)
                    # print("--- Predicted Results --- : \n ",pred_results)
                    print("--- Predicted Results --- : \n ")
                    print("Label Map : ",pred_results['label_map'])
                    print("Predicted Labels and confidences : ",pred_results['predicted_labels_and_confidences'])
                    dict_label = dict(pred_results['label_map'])
                    
                    for lbl, cnf in pred_results['predicted_labels_and_confidences']:
                        if cnf>=0.50:
                            enroll = dict_label[lbl]
                            if self.attendance_record[enroll]==0:
                                self.attendance_record[enroll]=1
                    print("Current Attandance Records: ",self.attendance_record)
                                    
                else:
                    print("No Face Found")
                    print("Current Attandance Records: ",self.attendance_record)    

                print("--------------- Image Processing Completed -----------------\n")

            except Exception as e:
                self.messageLbl.setText('Error: Snapshot capturing failed')
                print("Snapshot capturing failed...\n Errors:")
                print(e)


    def endClass(self):
        print("--------------- Class Ended -----------------")
        print("Final Attandance Records: ",self.attendance_record)
        if self.snapshotCnt>0:
            try:
                val1 = ''
                val2 = ''
                for enroll, present in self.attendance_record.items():
                    val1 = val1+"'"+ str(enroll)+"'" + ','
                    val2  = val2+str(present) + ','
                val1 = val1[:-1]
                val2 = val2[:-1]    
                format_str = """INSERT INTO %s (%s) VALUES (%s);""" % (current_subject_code,val1,val2)
                # print(format_str)
                      
                conn.execute(format_str)
                conn.commit()
                self.stopCapture()

            except Exception as e:
                    print("Error while exiting class...\n Errors:")
                    print(e)
        else:
            print("No snapshot taken so attendance not marked!")
        
        self.stopCapture()
        self.reset()
        win.setGeometry(win.frameGeometry())
        winTakeAttnGroupPhotos.hide()
        win.show()



class WindowShowAttnSelectClass(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Show Attendance Records")
        self.resize(window_width, window_height)
        self.move(QApplication.desktop().screen().rect().center()- self.rect().center())
        self.vbox = QVBoxLayout()
        self.subjectSelectLbl = QLabel('Select Subject')
        self.subjectSelect = QListWidget()

        # self.subjectSelect.setSortingEnabled(True)
      
        self.showButton = QPushButton("Show Attendance Records", self)
        self.showButton.clicked.connect(self.showAttn)
        self.backButton = QPushButton('Back')
        self.backButton.clicked.connect(self.back)
        self.hbox = QHBoxLayout()
        self.hbox.addWidget(self.showButton)
        self.hbox.addWidget(self.backButton)

        self.vbox.addWidget(self.subjectSelectLbl)
        self.vbox.addWidget(self.subjectSelect)
        self.vbox.addLayout(self.hbox)
        self.vbox.setAlignment(Qt.AlignCenter)
        self.setLayout(self.vbox)
        self.reset()
        

    def reset(self):
        self.subjectSelect.clear()
        sql_command = """SELECT * FROM subjects"""
        self.res = multiple_select_query(sql_command)
        if len(self.res)>0:
            for i in range(0,len(self.res)):
                item = QListWidgetItem(self.res[i][1] + ' (' + self.res[i][2]+ ') ')
                self.subjectSelect.addItem(item)
        
        self.subjectSelect.setCurrentRow(0)
        

    def showAttn(self):
        # Check which Class is selected!
        global current_subject_code
        # print(self.subjectSelect.currentRow())
        subject_name = self.subjectSelect.currentItem().text()
        subject_code = self.res[self.subjectSelect.currentRow()][2]
        print("Selected Subject: ",subject_name)
        
        current_subject_code = subject_code
        winShowAttnRecords.setGeometry(winShowAttnRecords.frameGeometry())
        winShowAttnSelectClass.hide()
        winShowAttnRecords.setWindowTitle("Attendance Records of " + subject_name + " class")
        winShowAttnRecords.showRecords()
        winShowAttnRecords.show()

       
        
    def back(self):
        win.setGeometry(win.frameGeometry())
        winShowAttnSelectClass.hide()
        win.show()


class WindowShowAttnRecords(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Attendance Records")
        self.resize(800, 400)
        self.move(QApplication.desktop().screen().rect().center()- self.rect().center())
        self.vbox = QVBoxLayout()
        self.vbox.setAlignment(Qt.AlignCenter)
        self.recordsTable = QTableWidget()
        
        self.hbox=QHBoxLayout()
        self.backButton = QPushButton('Back')
        self.backButton.setFixedWidth(400)
        self.backButton.clicked.connect(self.back)
        self.hbox.addWidget(self.backButton)

        self.vbox.setAlignment(Qt.AlignCenter)
        self.vbox.addWidget(self.recordsTable)
        self.vbox.addLayout(self.hbox)
        
        self.setLayout(self.vbox)
        
        

    def showRecords(self):
        try:
            sql_command1 = "SELECT * FROM %s" % (current_subject_code)
            res1 = multiple_select_query(sql_command1)
            sql_command2 = "SELECT enroll_no FROM reg_students WHERE subject_code = '%s' ORDER BY enroll_no" % (current_subject_code)
            enroll_list = select_query(sql_command2)
            sql_command2 = "SELECT student_name FROM reg_students WHERE subject_code = '%s' ORDER BY enroll_no" % (current_subject_code)
            name_list = select_query(sql_command2)

            lecture_count = len(res1)
            # print(res1,enroll_list)

            self.tableHeaders = ['Enrollment No','Student Name','Total Lectures','Present Count','Percent Attendance']
            # for i in range(0,lecture_count):
            #     self.tableHeaders.append('Lecture '+str(i+1))

            # self.tableHeaders.append('Present Count')
            # self.tableHeaders.append('Percent Attendance')



            self.recordsTable.setRowCount(len(enroll_list))
            # self.recordsTable.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            # self.recordsTable.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            self.recordsTable.setColumnCount(len(self.tableHeaders))
            self.recordsTable.setHorizontalHeaderLabels(self.tableHeaders)
            self.header = self.recordsTable.horizontalHeader()
            self.header.setResizeMode(QHeaderView.Stretch)
            # self.header.setStretchLastSection(True)
            self.recordsTable.setContentsMargins(5,5,5,5)

            for idx,enroll_no in enumerate(enroll_list):
                sql_command = "SELECT `%s` FROM %s" % (str(enroll_no),current_subject_code)
                attn = select_query(sql_command)
                present = 0
                for i in attn:
                    present=present+i
                attn_percent = (present/len(attn))*100
                str_attn_percent =   '%.2f' % attn_percent
                # print(sql_command)
                # print(attn)
                # row_content = [enroll_no]+attn+[present,str_attn_percent]
                row_content = [enroll_no,name_list[idx],len(attn),present,str_attn_percent]
                # print(row_content)
                for pos , item in enumerate(row_content):
                        table_item = QTableWidgetItem(str(item))
                        table_item.setTextAlignment(Qt.AlignCenter)
                        self.recordsTable.setItem(idx, pos , table_item)



        except Exception as e:
                print('Attendance records does not exist please take attendance first')
                print(e)

        


    def back(self):
        winShowAttnSelectClass.setGeometry(winShowAttnSelectClass.frameGeometry())
        winShowAttnRecords.hide()
        winShowAttnSelectClass.show()




class MainWindow:
    def __init__(self):
        self.win =  WindowMain()   
        self.winRegister = WindowRegister()
        self.winTakeAttnSelectClass = WindowTakeAttnSelectClass()
        self.winRegisterStudentDetails = WindowRegisterStudentDetails()
        self.winRegisterStudentPhotos = WindowRegisterStudentPhotos()
        self.winTakeAttnGroupPhotos = WindowTakeAttnGroupPhotos()
        self.winResults = WindowResults()
        self.win.show()        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win =  WindowMain()   
    winRegister = WindowRegister()
    winTakeAttnSelectClass = WindowTakeAttnSelectClass()
    winShowAttnSelectClass = WindowShowAttnSelectClass()
    winShowAttnRecords = WindowShowAttnRecords()
    winRegisterStudentDetails = WindowRegisterStudentDetails()
    winRegisterStudentPhotos = WindowRegisterStudentPhotos()
    winTakeAttnGroupPhotos = WindowTakeAttnGroupPhotos()
   
    win.show()        
    sys.exit(app.exec_())
