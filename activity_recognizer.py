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
        self.text.setText("")
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

    INACTIVE_TEXT = "the node is inactive. choose 'prediction'-mode to predict a gesture" \
                    " or 'training'-mode to train a new gesture"
    PREDICTION_TEXT = "press 'start' and execute a gesture." \
                      "press 'stop' when done and the category will be predicted"
    TRAINING_TEXT = "select an activtiy in the list and record performing the gesture" \
                    " by pressing the record button"

    state = GestureNodeState.INACTIVE
    gestures_dict = {}
    gesture_id = 0 

    def __init__(self, name):  
        Node.__init__(self, name, terminals={  
            'dataIn': {'io': 'in'},
            'prediction': {'io': 'out'},
            })
        self.prediction = '' 
        # example gestures- add more?
        self.saved_gestures = ['jump', 'run', 'throw'] 
        self.is_recording = False
        #svm
        self.svc = svm.SVC(kernel='rbf')
        self.init_ui()

    def init_ui(self):
        self.ui = QtGui.QWidget()
        self.layout = QtGui.QVBoxLayout()
        self.mode_layout = QtGui.QGridLayout()
        self.gesture_layout = QtGui.QGridLayout()
        self.pred_layout = QtGui.QGridLayout()
        self.gesture_widg = QtGui.QWidget()
        self.pred_widg = QtGui.QWidget()

        # init buttons for the mode: 'inactive', 'training' or 'predicting'
        self.init_mode_ui()

        # mode instructions
        self.mode_text_label = QtGui.QLabel()
        self.mode_text_label.setText(self.INACTIVE_TEXT)
        self.mode_layout.addWidget(self.mode_text_label, 1, 0, 3, 3)
        self.gesture_name = QtGui.QLineEdit()
        self.gesture_name.setVisible(False)
        self.mode_layout.addWidget(self.gesture_name, 7, 0, 2, 2)

        self.init_training_ui()
        self.init_prediction_ui()
        
        self.gesture_widg.setLayout(self.gesture_layout)
        self.pred_widg.setLayout(self.pred_layout)
        self.layout.addLayout(self.mode_layout)
        self.layout.addWidget(self.gesture_widg)
        self.layout.addWidget(self.pred_widg)
        self.pred_widg.setVisible(False)
        self.ui.setLayout(self.layout)

    def init_mode_ui(self):
        self.label_mode = QtGui.QLabel('select mode:')
        self.layout.addWidget(self.label_mode)

        self.inactive_button = QtWidgets.QRadioButton('inactive')
        self.inactive_button.setChecked(True)
    
        self.training_button = QtWidgets.QRadioButton('training')
        self.prediction_button = QtWidgets.QRadioButton('prediction')
        
        self.mode_layout.addWidget(self.inactive_button, 0,0)
        self.mode_layout.addWidget(self.training_button,0,1)
        self.mode_layout.addWidget(self.prediction_button,0,2)

        self.training_button.clicked.connect(lambda: self.on_mode_button_clicked(self.training_button))
        self.prediction_button.clicked.connect(lambda: self.on_mode_button_clicked(self.prediction_button))
        self.inactive_button.clicked.connect(lambda: self.on_mode_button_clicked(self.inactive_button))


    def init_training_ui(self):
        new_gesture_label = QtGui.QLabel("create a new gesture")
        self.gesture_layout.addWidget(new_gesture_label)
        self.gesture_name = QtGui.QLineEdit()
        self.gesture_layout.addWidget(self.gesture_name,7,0)

        self.gesture_select = QtWidgets.QComboBox()
        # self.gesture_select.addItems(self.saved_gestures)
        self.add_button = QtGui.QPushButton('add gesture')
        edit_gesture_label = QtGui.QLabel("edit gesture")
        self.train_button = QtGui.QPushButton('train gesture')
        self.delete_button = QtGui.QPushButton("delete gesture")
        self.record_button = QtGui.QPushButton("start recording")
        self.stop_record_button = QtGui.QPushButton("stop recording")

        self.gesture_layout.addWidget(self.add_button, 7, 1)
        self.gesture_layout.addWidget(edit_gesture_label)
        self.gesture_layout.addWidget(self.gesture_select, 9, 0)
        self.gesture_layout.addWidget(self.train_button, 9, 1)
        self.gesture_layout.addWidget(self.delete_button, 9, 2)
        self.gesture_layout.addWidget(self.record_button, 10, 1)
        self.gesture_layout.addWidget(self.stop_record_button, 10, 2)
        self.record_button.hide()
        self.stop_record_button.hide()

        self.add_button.clicked.connect(self.on_add_button_clicked)
        self.train_button.clicked.connect(self.on_train_button_clicked)
        self.delete_button.clicked.connect(self.on_delete_button_clicked)
        self.record_button.clicked.connect(self.on_record_button_clicked)
        self.stop_record_button.clicked.connect(self.on_stop_record_button_clicked)


    # prediction ui: start and stop button
    def init_prediction_ui(self):
        new_pred_label = QtGui.QLabel("predict a gesture")
        self.pred_layout.addWidget(new_pred_label)
        self.pred_start_button = QtGui.QPushButton("start predicting")
        self.pred_stop_button = QtGui.QPushButton("stop predicting")

        self.pred_layout.addWidget(self.pred_start_button, 11, 0)
        self.pred_layout.addWidget(self.pred_stop_button, 11, 1)

        self.pred_start_button.clicked.connect(self.on_pred_start_button_clicked)
        self.pred_stop_button.clicked.connect(self.on_pred_stop_button_clicked)


    # new gesture is added to the list
    def on_add_button_clicked(self):
        self.saved_gestures.append(self.gesture_name.text())
        self.gesture_select.addItem(self.saved_gestures[-1])
        #self.gestures_dict[self.gesture_id] = {}
        self.gestures_dict[self.gesture_name.text()] = []
        self.gesture_name.setText("")
        self.gesture_id += 1


    def on_train_button_clicked(self):
        self.record_button.show()
        self.stop_record_button.show()

    def on_record_button_clicked(self):
        self.activity_recording(True)
    
    def on_stop_record_button_clicked(self):
        self.activity_recording(False)
    
    def activity_recording(self, is_recording):
        self.is_recording = is_recording
        if self.state == GestureNodeState.TRAINING:
            if self.is_recording:
                self.mode_text_label.setText("Recording...")
            else:
                self.mode_text_label.setText(self.TRAINING_TEXT)


    def on_delete_button_clicked(self):      
        gesture_selected = self.gesture_select.currentText()
        self.saved_gestures.remove(gesture_selected)
        self.gesture_select.clear()
        self.gesture_select.addItems(self.saved_gestures)

    def on_pred_start_button_clicked(self):
        self.is_recording = True
        
    def on_pred_stop_button_clicked(self):
        self.is_recording = False
    

    def on_mode_button_clicked(self, buttonType):
        if buttonType is self.training_button:
            self.state = GestureNodeState.TRAINING
            self.mode_text_label.setText(self.TRAINING_TEXT)
            self.gesture_name.setText('')
            self.gesture_widg.setVisible(True)
            self.pred_widg.setVisible(False)
        elif buttonType is self.prediction_button:
            self.state = GestureNodeState.PREDICTING
            self.gesture_widg.setVisible(False)
            self.pred_widg.setVisible(True)
            self.mode_text_label.setText(self.PREDICTION_TEXT)
        else:
            self.state = GestureNodeState.INACTIVE
            self.pred_widg.setVisible(False)
            self.gesture_widg.setVisible(False)
            self.mode_text_label.setText(self.INACTIVE_TEXT)

    def ctrlWidget(self):  
        return self.ui

    def handle_gesture_training(self, kargs):
        if self.is_recording:
            input_val = kargs['dataIn']
            selected_gesture = self.gesture_select.currentText()
            print(self.gestures_dict)
            self.gestures_dict[selected_gesture].append(input_val)
        else:
            samples = []
            targets = []
        
            for key in self.gestures_dict:
                for feature in self.gestures_dict[key]:
                    feature = feature.flatten()
                    samples.append(feature)
                    targets.append(key)

            if not all(f == targets[0] for f in targets):
                self.svm.fit(samples, targets)
            

    def predict_gesture(self, kargs):
        if self.is_recording:
            try: 
                input_val = kargs['dataIn']
                self.prediction = self.svc.predict(input_val)
            except NotFittedError:
                return
            for key in self.gestures_dict:
                if key == prediction[0]:
                    return {'prediction': self.gestures_dict[key]}

    def process(self, **kargs):
        self.output = {'dataIn': "-"}
        if self.state == GestureNodeState.TRAINING:
            self.handle_gesture_training(kargs)

        if self.state == GestureNodeState.PREDICTING:
            self.output = self.predict_gesture(kargs)

        return self.output

fclib.registerNodeType(SvmNode, [('svm',)])


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
        self.frequency = None

    def process(self, **kargs):
        avg = []
        for i in range(len(kargs['accelX'])):
            avg.append((kargs['accelX'][i] + kargs['accelY'][i] + kargs['accelZ'][i]) / 3)
        self.frequency = np.abs(np.fft.fft(avg) / len(avg))[1:len(avg) // 2]
        return {'frequency': self.frequency}

fclib.registerNodeType(FftNode, [('fft',)])


def create_connect_nodes(chart):
    dippid_node = chart.createNode("DIPPID", pos=(0, 0))
    buffer_node_x = chart.createNode("Buffer", pos=(100, -200))
    buffer_node_y = chart.createNode("Buffer", pos=(100, 0))
    buffer_node_z = chart.createNode("Buffer", pos=(100, 200))
    fft_node = chart.createNode("fft", pos=(200, 100))
    display_node = chart.createNode("display", pos=(400,0))
    svm_node = chart.createNode("svm", pos=(300, 0))

    chart.connectTerminals(dippid_node['accelX'], buffer_node_x['dataIn'])
    chart.connectTerminals(dippid_node['accelY'], buffer_node_y['dataIn'])
    chart.connectTerminals(dippid_node['accelZ'], buffer_node_z['dataIn'])
    chart.connectTerminals(buffer_node_x['dataOut'], fft_node['accelX'])
    chart.connectTerminals(buffer_node_y['dataOut'], fft_node['accelY'])
    chart.connectTerminals(buffer_node_z['dataOut'], fft_node['accelZ'])
    chart.connectTerminals(fft_node['frequency'], svm_node['dataIn'])
    chart.connectTerminals(svm_node['prediction'], display_node['dataIn'])

def start():
    app = QtWidgets.QApplication([])

    win = QtGui.QMainWindow()
    win.resize(600,600)
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