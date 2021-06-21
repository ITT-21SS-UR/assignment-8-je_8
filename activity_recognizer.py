import sys
from enum import Enum
import pyqtgraph as pg
import numpy as np

from scipy.fft import fft
from sklearn import svm

from PyQt5 import QtWidgets
from pyqtgraph.Qt import QtGui, QtCore
from pyqtgraph.flowchart import Flowchart, Node
import pyqtgraph.flowchart.library as fclib

from DIPPID import SensorUDP, SensorSerial, SensorWiimote
from DIPPID_pyqtnode import BufferNode, DIPPIDNode

class GestureNodeState(Enum):
    TRAINING = 1
    PREDICTING = 2
    INACTIVE = 3

# implement a DisplayTextNode that displays the currently recognized/predicted category on the screen.
class DisplayTextNode(Node):
    nodeName = 'display'

    def __init__(self, name):  
        Node.__init__(self, name, terminals={  
            'dataIn': {'io': 'in'},
            'prediction': {'io': 'out'},
            })
        self.init_ui()
    
    def init_ui(self):
        self.ui = QtGui.QWidget()
        self.layout = QtGui.QGridLayout()
        predict_label = QtGui.QLabel("Prediction:")
        self.layout.addWidget(predict_label)
        self.text = QtGui.QLabel()
        self.port = "5700"
        self.text.setText(self.port)
        self.layout.addWidget(self.text)
        self.ui.setLayout(self.layout)
    
    def ctrlWidget(self):
        return self.ui

    def process(self, **kargs):
        prediction = kargs['dataIn'][0]
        self.text.setText(prediction)
        return {'prediction': self.text}




fclib.registerNodeType(DisplayTextNode, [('display',)])

        
# can be switched between training mode and prediction mode and "inactive" via buttons in the configuration pane. 
# in training mode it continually reads in a sample (i.e. a feature vector consisting of multiple values, 
# such as a list of frequency components) and trains a SVM classifier with this data (and previous data). 
# The category for this sample can be defined by a text field in the control pane.
# In prediction mode the SvmNode should read in a sample and output the predicted category as a string
class SvmNode(Node):
    nodeName = 'svm'

    state = GestureNodeState.INACTIVE

    def __init__(self, name):  
        Node.__init__(self, name, terminals={  
            'dataIn': {'io': 'in'},
            'prediction': {'io': 'out'},
            })
        self.prediction = '' 
        # example gestures- add more?
        self.saved_activities = ['jump', 'run', 'throw'] 
        self.is_recording = False
        #svm
        self.svc = svm.SVC()
        self.init_ui()

    def init_ui(self):
        self.ui = QtGui.QWidget()
        self.layout = QtGui.QVBoxLayout()
        self.mode_layout = QtGui.QGridLayout()
        self.activity_layout = QtGui.QGridLayout()

        # init buttons for the mode: 'inactive', 'training' or 'predicting'
        self.init_mode_ui()

        # mode instructions
        self.mode_text_label = QtGui.QLabel()
        self.mode_layout.addWidget(self.mode_text_label, 1, 0, 3, 3)
        self.activity_name = QtGui.QLineEdit()
        self.activity_name.setVisible(False)
        self.mode_layout.addWidget(self.activity_name, 7, 0, 2, 2)

        self.init_activity_ui()
        
        self.layout.addLayout(self.mode_layout)
        self.layout.addLayout(self.activity_layout)
        self.ui.setLayout(self.layout)

    def init_mode_ui(self):
        self.label_mode = QtGui.QLabel('select mode:')
        self.layout.addWidget(self.label_mode)

        self.inactive_button = QtWidgets.QRadioButton('inactive')
        self.training_button = QtWidgets.QRadioButton('training')
        self.prediction_button = QtWidgets.QRadioButton('prediction')
        
        self.mode_layout.addWidget(self.inactive_button, 0,0)
        self.mode_layout.addWidget(self.training_button,0,1)
        self.mode_layout.addWidget(self.prediction_button,0,2)

        self.training_button.clicked.connect(lambda: self.on_mode_button_clicked(self.training_button))
        self.prediction_button.clicked.connect(lambda: self.on_mode_button_clicked(self.prediction_button))
        self.inactive_button.clicked.connect(lambda: self.on_mode_button_clicked(self.inactive_button))


    def init_activity_ui(self):
        new_activity_label = QtGui.QLabel("create a new activity")
        self.activity_layout.addWidget(new_activity_label)
        self.activity_name = QtGui.QLineEdit()
        self.activity_layout.addWidget(self.activity_name,7,0)

        self.activity_select = QtWidgets.QComboBox()
        self.activity_select.addItems(self.saved_activities)
        self.add_button = QtGui.QPushButton('add actvity')
        edit_activity_label = QtGui.QLabel("edit activity")
        self.retrain_button = QtGui.QPushButton('retrain activity')
        self.delete_button = QtGui.QPushButton("delete activity")

        self.activity_layout.addWidget(self.add_button, 7,1)
        self.activity_layout.addWidget(edit_activity_label)
        self.activity_layout.addWidget(self.activity_select,9,0)
        self.activity_layout.addWidget(self.retrain_button,9,1)
        self.activity_layout.addWidget(self.delete_button,9,2)

        self.add_button.clicked.connect(self.on_add_button_clicked)
        self.retrain_button.clicked.connect(self.on_retrain_button_clicked)
        self.delete_button.clicked.connect(self.on_delete_button_clicked)

    # training ui: start and stop recording button
    # hide other widgets from activity_layout ?
    def init_training_ui(self):
        pass

    #prediction ui: start and stop button
    def init_prediction_ui(self):
        pass

    # new activity is added to the list
    def on_add_button_clicked(self):
        self.saved_activities.append(self.activity_name.text())
        self.activity_name.setText("")
        self.activity_select.addItem(self.saved_activities[-1])

    def on_retrain_button_clicked(self):
        print("retrain button clicked")

    def on_delete_button_clicked(self):      
        print("delete button clicked")

    def on_mode_button_clicked(self, buttonType):
        if buttonType is self.training_button:
            self.state = GestureNodeState.TRAINING
            self.mode_text_label.setText("select an activtiy in the list and record performing the activity" 
                                                "by pressing the record button")
            self.activity_name.setText('')
            self.activity_name.setVisible(True)
            self.init_training_ui()

        elif buttonType is self.prediction_button:
            self.state = GestureNodeState.PREDICTING
            self.mode_text_label.setText("press 'button 1' and execute an acitivity. after releasing it " 
                                            " it will predict your activity")
            self.activity_name.setVisible(False)
            self.init_prediction_ui()

        else:
            self.state = GestureNodeState.INACTIVE
            self.mode_text_label.setText("the node is inactive. choose 'prediction'-mode to predict a activity"  
                                            " or 'training'-mode to train a new activity")
            self.activity_name.setVisible(False)

    def ctrlWidget(self):  
        return self.ui

    def predict_activity(self, kargs):
        self.prediction = self.svc.predict(kargs['dataIn'])
        print(self.prediction)

    def process(self, **kargs):
        return {'frequency': self.predict}

fclib.registerNodeType(SvmNode, [('fft',)])


# reads in information from a BufferNode and outputs a frequency spectrogram
class FftNode(Node):
    nodeName = 'fft'

    def __init__(self, name):  
        Node.__init__(self, name, terminals={  
            'accelX': {'io': 'in'},
            'accelY': {'io': 'in'},
            'accelZ': {'io': 'in'},
            'frequency': {'io': 'out'},
            })

    def process(self, **kwds):
        # kwds will have one keyword argument per input terminal.

        return {'freqency': frequency}

fclib.registerNodeType(FftNode, [('fft',)])


def create_connect_nodes(chart):
    dippid_node = chart.createNode("DIPPID", pos=(0, 0))
    buffer_node_x = chart.createNode("Buffer", pos=(100, -200))
    buffer_node_y = chart.createNode("Buffer", pos=(100, 0))
    buffer_node_z = chart.createNode("Buffer", pos=(100, 200))
    fft_node = chart.createNode("fft", pos=(200, 100))
    svm_node = chart.createNode("svm", pos=(300, 0))
    display_node = chart.createNode("display", pos=(400,0))

    chart.connectTerminals(dippid_node['accelX'], buffer_node_x['dataIn'])
    chart.connectTerminals(dippid_node['accelY'], buffer_node_y['dataIn'])
    chart.connectTerminals(dippid_node['accelZ'], buffer_node_z['dataIn'])
    chart.connectTerminals(buffer_node_x['dataOut'], fft_node['accelX'])
    chart.connectTerminals(buffer_node_y['dataOut'], fft_node['accelY'])
    chart.connectTerminals(buffer_node_z['dataOut'], fft_node['accelZ'])

def start():
    app = QtWidgets.QApplication([])

    win = QtGui.QMainWindow()
    win.setWindowTitle("Activity Recognizer")
    central_widget = QtGui.QWidget()
    win.setCentralWidget(central_widget)

    layout = QtGui.QGridLayout()
    central_widget.setLayout(layout)
    fc = Flowchart(terminals={'out': dict(io='out')})
    layout.addWidget(fc.widget(), 0, 0, 2, 1)
    
    create_connect_nodes(fc)

    win.show()
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        sys.exit(QtGui.QApplication.instance().exec_())


if __name__ == '__main__':
    start()