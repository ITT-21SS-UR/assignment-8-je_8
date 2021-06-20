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
    nodeName = 'text'


fclib.registerNodeType(DisplayTextNode, [('text',)])

        
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
        self.gestures = ['jump', 'run', 'throw'] 
        self.init_gui()

    def init_gui(self):
        self.gui = QtGui.QWidget()
        self.layout = QtGui.QVBoxLayout()
        self.mode_layout = QtGui.QGridLayout()

        # init buttons
        self.init_buttons()
        self.init_mode_buttons()

        self.layout.addWidget(self.add_button)
        self.layout.addWidget(self.edit_button)
        self.layout.addWidget(self.delete_button)

        self.gui.setLayout(self.layout)

    def init_buttons(self):
        self.add_button = QtGui.QPushButton('add gesture')
        self.edit_button = QtGui.QPushButton('edit gesture')
        self.delete_button = QtGui.QPushButton("delete gesture")

        self.add_button.clicked.connect(self.on_add_button_clicked)
        self.edit_button.clicked.connect(self.on_edit_button_clicked)
        self.delete_button.clicked.connect(self.on_delete_button_clicked)

    def init_mode_buttons(self):
        self.label_mode = QtGui.QLabel('select mode:')
        self.layout.addWidget(self.label_mode)

        self.inactive_button = QtWidgets.QRadioButton('inactive')
        self.layout.addWidget(self.inactive_button)

        self.training_button = QtWidgets.QRadioButton('training')
        self.layout.addWidget(self.training_button)

        self.prediction_button = QtWidgets.QRadioButton('prediction')
        self.layout.addWidget(self.prediction_button)

        self.training_button.clicked.connect(lambda: self.on_mode_button_clicked(self.training_button))
        self.prediction_button.clicked.connect(lambda: self.on_mode_button_clicked(self.prediction_button))
        self.inactive_button.clicked.connect(lambda: self.on_mode_button_clicked(self.inactive_button))

    def on_add_button_clicked(self):
        print("add button clicked")

    def on_edit_button_clicked(self):
        print("edit button clicked")

    def on_delete_button_clicked(self):      
        print("delete button clicked")

    def on_mode_button_clicked(self, buttonType):
        if buttonType is self.training_button:
            self.state = GestureNodeState.TRAINING
            self.train_help_label.setText("Select a gesture in the list and record performing the gesture" \
                                                "by pressing the record button")

        elif buttonType is self.prediction_button:
            self.state = GestureNodeState.PREDICTING
            self.train_help_label.setText("")

        else:
            self.state = GestureNodeState.INACTIVE
            self.train_help_label.setText("")



    def ctrlWidget(self):  
        return self.gui

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
    display_node = chart.createNode("text", pos=(400,0))

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