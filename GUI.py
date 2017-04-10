import os
import sys
import csv
import cv2
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
    res =[x[0].encode('utf8') for x in res]
    return res

def int_select_query(comm,params=()):
    cursor.execute(comm,params)
    res = cursor.fetchall()
    res =[x[0] for x in res]
    return res



class WindowMain(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("SAMs - Smart Attendance Management system")
        self.resize(window_width, window_height)
        self.centerWindow()
        self.vbox = QVBoxLayout()
        self.vbox.setAlignment(Qt.AlignCenter)

        self.button1 = QPushButton("Register", self)
        self.button1.clicked.connect(self.onclick_button1)

        self.button2 = QPushButton("Take Attendance", self)
        self.button2.clicked.connect(self.onclick_button2)

        self.vbox.addWidget(self.button1)
        self.vbox.addWidget(self.button2)
        self.setLayout(self.vbox)

    def onclick_button1(self):
        # Register the student
        win.hide()
        winRegister.show()

    def onclick_button2(self):
        # Take Attendance
        win.hide()
        winTakeAttnSelectClass.show()

    def centerWindow(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


class WindowRegister(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Register")
        self.resize(window_width, window_height)
        self.centerWindow()
        self.fbox = QFormLayout()

        self.textLabel1 = QLabel("Subject Name: ", self)
        self.textLabel2 = QLabel("Subject Code: ", self)
        self.textLabel3 = QLabel("No. of Students: ", self)
        self.messageLbl = QLabel('')
        
        self.lineEdit1 = QLineEdit(self)
        self.lineEdit2 = QLineEdit(self)
        self.lineEdit3 = QLineEdit(self)

        self.button1 = QPushButton("Start", self)
        self.button1.clicked.connect(self.onclick_button1)

        self.fbox.addRow(self.textLabel1,self.lineEdit1)
        self.fbox.addRow(self.textLabel2,self.lineEdit2)
        self.fbox.addRow(self.textLabel3,self.lineEdit3)
        self.fbox.addRow(self.messageLbl)
        self.fbox.addRow(self.button1)
        self.fbox.setAlignment(Qt.AlignCenter)
        self.fbox.setContentsMargins(100, 50, 100, 50)
        self.fbox.setSpacing(10)
        self.setLayout(self.fbox)

    def onclick_button1(self):
        # Start the Registration process
        if (not self.lineEdit1.text()) or (not self.lineEdit2.text()) or (not self.lineEdit3.text()):
            self.messageLbl.setText('Error: One or more required fields empty! verification failed')
            return
        else:
            subject_name = str(self.lineEdit1.text())
            subject_code = str(self.lineEdit2.text())
            no_of_students = str(self.lineEdit3.text())
            print(subject_name,subject_code,no_of_students)
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
            
        winRegister.hide()
        self.lineEdit1.clear()
        self.lineEdit2.clear()
        winRegisterStudentDetails.show()

    def centerWindow(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


class WindowRegisterStudentDetails(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Enter Student Details")
        self.resize(window_width, window_height)
        self.centerWindow()
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
            print(enroll_no,student_name)
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
        winRegisterStudentDetails.hide()
        winRegisterStudentPhotos.show()

    def centerWindow(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

class WindowRegisterStudentPhotos(QWidget):
    def __init__(self):
        super().__init__()
        self.capturing=False
        self.setFixedSize(700,460)
        self.centerWindow()
        self.video_size = QSize(400, 300)
        self.snapshot_size = QSize(80, 80)
        self.cascPath = 'haarcascade_frontalface_default.xml'
        self.faceCascade = cv2.CascadeClassifier(self.cascPath)
        self.snapshotCnt=0
        self.maxSnapshotCnt=8
        self.captureCompleted = False
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
        self.hbox.addWidget(self.trainModelButton)
        
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

        winRegisterStudentPhotos.hide()
        winRegisterStudentDetails.show()

    def trainModel(self):
                #MY GLOBAL CONSTANTS
        LOAD_PATH= os.path.join(image_path,current_subject_code)
        #STORE_PATH=store_path#store data in this folder
        DSETP=dataset_path
        MODLP=model_path

        DSETFN=current_subject_code
        MFN="model_"+DSETFN# model filename

        NEPOCH=10
        DBS=500# data batch size
        TBS=10# training batch size
        DMY="no val"# dummmy

        #load_path: 
        #DSETP : dataset path where you want to save. You will pass this path. f
        dsop.createSingleBlockDataset(LOAD_PATH,DSETP,DSETFN,(50,50,3))
        md=dsop.loadMetaData(DSETP+'/'+DSETFN+'_metadata.txt')
        print(md[0],md[0]["shape"])

        print("PATH:",DSETP+"/"+DSETFN+".h5",md[0]["shape"])
        #dsop.navigateDataset(DSETP+"/"+DSETFN+".h5",md[0]["shape"],0)

        dsop.partitionDataset(DSETP+"/"+DSETFN+".h5",DSETP+"/"+DSETFN+"_metadata.txt",(80,20))

        md=dsop.loadMetaData(DSETP+'/'+DSETFN+'_train_metadata.txt')
        model= cmg.getModelFrame(md[0]["shape"],int(md[0]["nb_classes"]),3)
        DBS= md[0]["dataset_shape"][0]
        MFN=MFN+"_"+str(NEPOCH)
        #model_path, model_md_path=getTrainedModel(model,DSETP+"/"+DSETFN+"_train.h5",DSETP+"/"+DSETFN+"_train_metadata.txt",
        #                              MODLP,MFN,NEPOCH,DBS,TBS)


        MODEL_LOC=MODLP+"/"+MFN+".h5"
        TD_LOC=DSETP+"/"+DSETFN+"_test.h5"
        TD_MD_LOC=DSETP+"/"+DSETFN+"_test_metadata.txt"


        # tu ye use kr lena for training the model so that you can use it for prediction later.
        #It will return the path of the best model. Though the model will be saved at othercheck points also(e.g. here
        #once the accuracy reaches 70 )
        trained_model_path, model_md_path=cmg.getCustomOptimalTrainedModel(model,DSETP+"/"+DSETFN+"_train.h5",
                                                                   DSETP+"/"+DSETFN+"_train_metadata.txt",
                                                             MODLP,MFN,70,2,
                                                             0.8,15,0.2,TD_LOC,TD_MD_LOC,10000)
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


        # img=dsop.cv.imread('2.jpg')

        # pred_results = cmg.labelFaces(trained_model_path,model_md_path,img)
        # print ("Prediction_results:\n",pred_results)
        # dsop.cv.imshow("Img",pred_results["image"])
        # dsop.cv.waitKey(0)

        
            

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

    def centerWindow(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())



class WindowTakeAttnSelectClass(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Select Subject")
        self.resize(600, 250)
        self.centerWindow()
        self.vbox = QVBoxLayout()
        self.subjectSelect = QListWidget()
      
        self.startButton = QPushButton("Start", self)
        self.startButton.clicked.connect(self.startClass)
        self.vbox.addWidget(self.subjectSelect)
        self.vbox.addWidget(self.startButton)
        self.vbox.setAlignment(Qt.AlignCenter)
        self.setLayout(self.vbox)
        sql_command = """SELECT * FROM subjects"""
        self.res = multiple_select_query(sql_command)
        # print(res)
        if len(self.res)>0:
            for i in range(0,len(self.res)):
                # print(res[i][1])
                self.subjectSelect.addItem(self.res[i][1])




    def startClass(self):
        # Check which Class is selected!
        global current_subject_code
        global current_model_path
        global current_model_md_path
        # print(self.subjectSelect.currentRow())
        subject_name = self.subjectSelect.currentItem().text()
        subject_code = self.res[self.subjectSelect.currentRow()][2]
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
       
        sql_command = "SELECT enroll_no FROM reg_students WHERE subject_code = '%s' " % (current_subject_code)        
        enroll_list = int_select_query(sql_command)
        col = ""
        for i in enroll_list:
            col+=",\n'%s' INTEGER NOT NULL DEFAULT 0" % (str(i))
        
        sql_command = """CREATE TABLE IF NOT EXISTS %s (
                    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                    date_and_time DATETIME NOT NULL DEFAULT (datetime('now','localtime'))%s
                    );""" %(current_subject_code,col)
        
        print(sql_command)            
        query(sql_command)            
        
        winTakeAttnSelectClass.hide()
        winTakeAttnGroupPhotos.show()

        winTakeAttnGroupPhotos.setWindowTitle("Take Attendance Photos for " + demoVar)
        

    def centerWindow(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


class WindowTakeAttnGroupPhotos(QWidget):
    def __init__(self):
        super().__init__()
        self.capturing=False
        self.setFixedSize(700,460)
        self.centerWindow()
        self.video_size = QSize(400, 300)
        self.snapshot_size = QSize(80, 80)
        self.cascPath = 'haarcascade_frontalface_default.xml'
        self.faceCascade = cv2.CascadeClassifier(self.cascPath)
        self.snapshotCnt=0
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
            if not hasattr(self,'attendance_record'):
                self.attendance_record = {}
                sql_command = "SELECT enroll_no FROM reg_students WHERE subject_code = '%s' " % (current_subject_code)
                # cursor.execute(sql_command)
                # res = cursor.fetchall()
                res = int_select_query(sql_command)
                if len(res)>0:
                    print(res)
                    for i in res:
                        self.attendance_record[str(i)]=0

                print(self.attendance_record)
            self.capture = cv2.VideoCapture(camera_port)
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
                if len(faces)>0:
                    self.snapshotCnt=self.snapshotCnt+1
                    pred_results=cmg.labelFaces(current_model_path,current_model_md_path,frame)
                    # print("--- Predicted Results --- : \n ",pred_results)
                    print(" Label_Map : ",pred_results['label_map'])
                    print(" predicted_labels_and_confidences : ",pred_results['predicted_labels_and_confidences'])
                    dict_label = dict(pred_results['label_map'])
                    
                    for lbl, cnf in pred_results['predicted_labels_and_confidences']:
                        if cnf>=0.50:
                            enroll = dict_label[lbl]
                            if self.attendance_record[enroll]==0:
                                self.attendance_record[enroll]=1
                    print(self.attendance_record)
                    

            except Exception as e:
                self.messageLbl.setText('Error: Snapshot capturing failed')
                print("Snapshot capturing failed...\n Errors:")
                print(e)


    def endClass(self):
        val1 = ''
        val2 = ''
        for enroll, present in self.attendance_record.items():
            val1 = val1+"'"+ str(enroll)+"'" + ','
            val2  = val2+str(present) + ','
        val1 = val1[:-1]
        val2 = val2[:-1]    
        format_str = """INSERT INTO %s (%s) VALUES (%s);""" % (current_subject_code,val1,val2)
        print(format_str)
              
        conn.execute(format_str)
        conn.commit()
        self.stopCapture()
        self.snapshotCnt=0

        # end class


    def centerWindow(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

class WindowResults(QtGui.QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Attendance Results")
        self.resize(600, 250)
        self.centerWindow()

    def centerWindow(self):
        qr = self.frameGeometry()
        cp = QtGui.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

class MainWindow:
    def __init__(self):
        self.win =  WindowMain()   
        self.winRegister = WindowRegister()
        self.winTakeAttnSelectClass = WindowTakeAttnSelectClass()
        self.winRegisterStudentDetails = WindowRegisterStudentDetails()
        self.winRegisterStudentPhotos = WindowRegisterStudentPhotos()
        self.winTakeAttnGroupPhotos = WindowTakeAttnGroupPhotos()
        self.winResults = WindowResults()
        self.win.setStyleSheet("#gframe {border-radius:5px;border:1px solid #a5a5a5}")
        self.win.show()        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win =  WindowMain()   
    winRegister = WindowRegister()
    winTakeAttnSelectClass = WindowTakeAttnSelectClass()
    winRegisterStudentDetails = WindowRegisterStudentDetails()
    winRegisterStudentPhotos = WindowRegisterStudentPhotos()
    winTakeAttnGroupPhotos = WindowTakeAttnGroupPhotos()
    winResults = WindowResults()
    win.show()        
    sys.exit(app.exec_())
