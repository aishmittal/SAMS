"""
@author: SKS

Prototype UI for SAMs.

"""

import os
import sys
import csv
from PyQt4 import QtGui, QtCore

# Global Variables
demoVar = "Physics Class"


class WindowMain(QtGui.QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("SAMs - Smart Attendance Management system")
        self.resize(600, 250)
        self.centerWindow()

        button1 = QtGui.QPushButton("Register", self)
        button1.move(250, 100)
        button1.clicked.connect(self.onclick_button1)

        button2 = QtGui.QPushButton("Take Attendance", self)
        button2.move(250, 150)
        button2.clicked.connect(self.onclick_button2)

    def onclick_button1(self):
        # Register the student
        # win.hide()
        winRegister.show()

    def onclick_button2(self):
        # Take Attendance
        # win.hide()
        winTakeAttnSelectClass.show()

    def centerWindow(self):
        qr = self.frameGeometry()
        cp = QtGui.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


class WindowRegister(QtGui.QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Register")
        self.resize(600, 250)
        self.centerWindow()

        textLabel1 = QtGui.QLabel("No. of Students: ", self)
        textLabel1.move(100, 80)

        textLabel2 = QtGui.QLabel("Class Name: ", self)
        textLabel2.move(100, 130)

        self.lineEdit1 = QtGui.QLineEdit(self)
        self.lineEdit1.move(200, 80)

        self.lineEdit2 = QtGui.QLineEdit(self)
        self.lineEdit2.move(200, 130)

        button1 = QtGui.QPushButton("Start", self)
        button1.move(450, 200)
        button1.clicked.connect(self.onclick_button1)

    def onclick_button1(self):
        # Start the Registration process
        winRegister.hide()
        self.lineEdit1.clear()
        self.lineEdit2.clear()
        winRegisterStudentDetails.show()

    def centerWindow(self):
        qr = self.frameGeometry()
        cp = QtGui.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


class WindowRegisterStudentDetails(QtGui.QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Enter Student Details")
        self.resize(600, 250)
        self.centerWindow()

        textLabel1 = QtGui.QLabel("Enrollment No: ", self)
        textLabel1.move(100, 80)

        textLabel2 = QtGui.QLabel("Student Name: ", self)
        textLabel2.move(100, 130)

        self.lineEdit1 = QtGui.QLineEdit(self)
        self.lineEdit1.move(200, 80)

        self.lineEdit2 = QtGui.QLineEdit(self)
        self.lineEdit2.move(200, 130)
        self.lineEdit2.resize(250, 30)

        button1 = QtGui.QPushButton("Next", self)
        button1.move(450, 200)
        button1.clicked.connect(self.onclick_button1)

    def onclick_button1(self):
        # Forward to next page
        winRegisterStudentDetails.hide()
        self.lineEdit1.clear()
        self.lineEdit2.clear()

        winRegisterStudentPhotos.show()

    def centerWindow(self):
        qr = self.frameGeometry()
        cp = QtGui.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


class WindowRegisterStudentPhotos(QtGui.QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Take Photographs")
        self.resize(600, 250)
        self.centerWindow()

        textLabel1 = QtGui.QLabel("", self)
        textLabel1.resize(220, 220)
        textLabel1.move(100, 20)
        textLabel1.setPixmap(QtGui.QPixmap(os.getcwd() + "/1.jpg"))

        button1 = QtGui.QPushButton("Start", self)
        button1.move(450, 150)
        button1.clicked.connect(self.onclick_button1)

        button2 = QtGui.QPushButton("Stop", self)
        button2.move(450, 200)
        button2.clicked.connect(self.onclick_button2)

    def onclick_button1(self):
        # Start taking photos for Database
        pass

    def onclick_button2(self):
        # Stop taking photos
        self.showCompletionMsg()

    def centerWindow(self):
        qr = self.frameGeometry()
        cp = QtGui.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def showCompletionMsg(self):
        reply = QtGui.QMessageBox.question(self, "Done!", "Add data for another Student?",
                                           QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
        if reply == QtGui.QMessageBox.No:
            # Go to main page
            winRegisterStudentPhotos.hide()
        else:
            winRegisterStudentPhotos.hide()
            winRegisterStudentDetails.show()


class WindowTakeAttnSelectClass(QtGui.QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Take Attendance")
        self.resize(600, 250)
        self.centerWindow()

        self.listWidget = QtGui.QListWidget(self)
        self.listWidget.addItem("Physics Class")
        self.listWidget.addItem("Biology Class")
        self.listWidget.move(150, 100)
        self.listWidget.resize(200, 100)

        button1 = QtGui.QPushButton("Start", self)
        button1.move(450, 200)
        button1.clicked.connect(self.onclick_button1)

    def onclick_button1(self):
        # Check which Class is selected!
        winTakeAttnSelectClass.hide()
        winTakeAttnGroupPhotos.show()

        demoVar = self.listWidget.currentItem().text()
        winTakeAttnGroupPhotos.setWindowTitle("Take Attendance Photos for " + demoVar)
        print(self.listWidget.currentItem().text())

    def centerWindow(self):
        qr = self.frameGeometry()
        cp = QtGui.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


class WindowTakeAttnGroupPhotos(QtGui.QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Take Attendance Photos for " + demoVar)
        self.resize(600, 250)
        self.centerWindow()

        textLabel1 = QtGui.QLabel("", self)
        textLabel1.resize(300, 220)
        textLabel1.move(50, 20)
        textLabel1.setPixmap(QtGui.QPixmap(os.getcwd() + "/2.jpg"))

        button1 = QtGui.QPushButton("Take Photo", self)
        button1.move(450, 150)
        button1.clicked.connect(self.onclick_button1)

        button2 = QtGui.QPushButton("Done", self)
        button2.move(450, 200)
        button2.clicked.connect(self.onclick_button2)

    def onclick_button1(self):
        # Just click a photo
        pass

    def onclick_button2(self):
        # Done taking photos
        winTakeAttnGroupPhotos.hide()
        winResults.show()

    def centerWindow(self):
        qr = self.frameGeometry()
        cp = QtGui.QDesktopWidget().availableGeometry().center()
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


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)

    win = WindowMain()
    win.show()

    winRegister = WindowRegister()
    winTakeAttnSelectClass = WindowTakeAttnSelectClass()
    winRegisterStudentDetails = WindowRegisterStudentDetails()
    winRegisterStudentPhotos = WindowRegisterStudentPhotos()
    winTakeAttnGroupPhotos = WindowTakeAttnGroupPhotos()
    winResults = WindowResults()

    sys.exit(app.exec_())
