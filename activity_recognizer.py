import sys
from enum import Enum
import pyqtgraph as pg
import numpy as np
from scipy.fft import fft

from PyQt5 import QtWidgets
from pyqtgraph.Qt import QtGui, QtCore
from pyqtgraph.flowchart import Flowchart
import pyqtgraph.flowchart.library as fclib

from DIPPID import SensorUDP, SensorSerial, SensorWiimote
from DIPPID_pyqtnode import BufferNode, DIPPIDNode

class GestureNodeState(Enum):
    TRAINING = 1
    PREDICTING = 2
    INACTIVE = 3

# implement a DisplayTextNode that displays the currently recognized/predicted category on the screen.
class DisplayTextNode(QtWidgets.QWidget):
    nodeName = 'text'


fclib.registerNodeType(DisplayTextNode, [('text',)])

        
# can be switched between training mode and prediction mode and "inactive" via buttons in the configuration pane. 
# in training mode it continually reads in a sample (i.e. a feature vector consisting of multiple values, 
# such as a list of frequency components) and trains a SVM classifier with this data (and previous data). 
# The category for this sample can be defined by a text field in the control pane.
# In prediction mode the SvmNode should read in a sample and output the predicted category as a string
class SvmNode(Node):
    nodeName = 'svm'

    def __init__(self, name):  
        Node.__init__(self, name, terminals={  
            'dataIn': {'io': 'in'},
            'prediction': {'io': 'out'},
            })
        self.prediction = '' 
        # example gestures- add more?
        self.gestures = ['jump', 'run', 'throw'] 
        self.init_gui()

    def.init_gui(self):
        self.gui = QtGui.QWidget()
        self.layout = QtGui.QVBoxLayout()
        self.mode_layout = QtGui.QGridLayout()

        # init buttons




    def ctrlWidget(self):  
        return self.gui

    def process(self, **kargs):
        return {'frequency': self.predict}

fclib.registerNodeType(FftNode, [('fft',)])


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
    chart.connectTerminals(buffer_node_x['dataOut'], fft_node['in_x'])
    chart.connectTerminals(buffer_node_y['dataOut'], fft_node['in_y'])
    chart.connectTerminals(buffer_node_z['dataOut'], fft_node['in_z'])

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


if __name__ == '__main__':
    start()