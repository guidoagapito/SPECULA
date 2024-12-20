
import queue
import pickle

from specula import cpuArray
from specula.base_processing_obj import BaseProcessingObj
from specula.base_value import BaseValue
from specula.data_objects.intensity import Intensity
from specula.data_objects.pixels import Pixels
from specula.data_objects.slopes import Slopes
from specula.data_objects.ef import ElectricField
from contextlib import contextmanager


def remove_xp_np(obj):
    '''Remove any instance of xp and np modules
    and return the removed modules
    
    Removed modules are returned separately, so that
    they can avoid the pickle stage
    
    Works recursively on object lists
    '''
    attrnames = ['xp', 'np']
    if isinstance(obj, list):
        return list(map(remove_xp_np, obj))

    deleted = {}
    for attrname in attrnames:
        if hasattr(obj, attrname):
            deleted[attrname] = getattr(obj, attrname)
            delattr(obj, attrname)
    return deleted


def putback_xp_np(obj, deleted):
    '''Put back the removed modules, if any.

    Works recursively on object lists
    '''
    if isinstance(obj, list):
        _ = list(map(putback_xp_np, zip(obj, deleted)))

    for k, v in deleted:
        setattr(obj, k, v)
       

class ProcessingDisplay(BaseProcessingObj):
    def __init__(self, qin, qout, data_obj_getter):
        super().__init__()
        self.qin = qin
        self.qout = qout
        self.data_obj_getter = data_obj_getter
        
    def trigger(self):
        while True:
            try:
                request = self.qin.get(block=False)
            except queue.Empty:
                return

            if request is None:
                self.qout.put((None, None))
                continue

            dataobj = self.data_obj_getter(request)
            dataobj_cpu = dataobj.copyTo(-1)

            deleted = remove_xp_np(dataobj_cpu)

            # Double pickle trick (we pickle, and then qout will pickle again)
            # to avoid some problems
            # with serialization of modules, which apparently
            # are still present even after the xp and np removal,
            # and only in input objects.                
            obj_bytes = pickle.dumps(dataobj_cpu)
            self.qout.put((request, obj_bytes))

            # Put xp and np back in case this was a reference
            # and not a real copy            
            putback_xp_np(dataobj_cpu, deleted)
            
        
from flask import Flask, render_template
from flask_socketio import SocketIO

import numpy as np
import io
import time
import base64
import matplotlib
matplotlib.use('Agg') # Memory backend, no GUI
from matplotlib.figure import Figure

app = Flask('Specula_display_server')
socketio = SocketIO(app)

server = None

def encode(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    imgB64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return imgB64


qin = None
qout = None
plotters = {}

class Plotter():
    def __init__(self, disp_factor=1, histlen=200, wsize=(600, 400), window=23, yrange=(-10, 10), oplot=False, color=1, psym=-4, title=''):
        super().__init__()
        
        self._wsize = wsize
        self._history = np.zeros(histlen)
        self._count = 0
        self._yrange = yrange
        self._value = None
        self._oplot = oplot
        self._color = color
        self._psym = psym
        self._title = title
        self._opened = False
        self._first = True
        self._disp_factor = disp_factor
        
    def set_w(self, size_frame=None):
        if size_frame is None:
            size_frame = self._wsize
        self.fig = Figure(figsize=(size_frame[0] * self._disp_factor / 100, size_frame[1] * self._disp_factor / 100))
        self.ax = self.fig.add_subplot(111)

    def plot(self,  objref):

        if isinstance(objref, Intensity):
            return self.imshow(objref.i)

        elif isinstance(objref, Pixels):
            return self.imshow(objref.pixels)

        elif isinstance(objref, Slopes):
            x, y = objref.get2d()
            return self.imshow(np.hstack((x, y)))

        elif isinstance(objref, BaseValue):
            value = objref._value
            if value is None:
                return self.plot_vector([0,0])
            
            if len(value.shape) == 2:
                return self.imshow(value)

            elif len(value.shape) == 1 and len(value) > 1:
                return self.plot_vector(value)

            # Scalar value: plot history
            else:  
                n = len(self._history)
                if self._count >= n:
                    self._history[:-1] = self._history[1:]
                    self._count = n - 1

                if not self._opened:
                    self.set_w()
                    self._opened = True

                self._history[self._count] = value
                self._count += 1

                x = np.arange(self._count)
                y = self._history[:self._count]
                if self._first:
                    self.fig.suptitle(self._title)
                    self.line = self.ax.plot(x, y, marker='.')
                    self._first = False
                else:
                    self.line[0].set_xdata(x)
                    self.line[0].set_ydata(y)
                    self.ax.set_xlim(x.min(), x.max())
                    if np.sum(np.abs(self._yrange)) > 0:
                        self.ax.set_ylim(self._yrange[0], self._yrange[1])
                    else:
                        self.ax.set_ylim(y.min(), y.max())
                self.fig.canvas.draw()
                return self.fig
        
        elif isinstance(objref, ElectricField):
            phase = objref

            frame = cpuArray(phase.phaseInNm * (phase.A > 0).astype(float))
            idx = np.where(cpuArray(phase.A) > 0)[0]

            # Remove average phase
            frame[idx] -= np.mean(frame[idx])

            if np.sum(self._wsize) == 0:
                size_frame = frame.shape
            else:
                size_frame = self._wsize
            return self.imshow(frame)
                
    def plot_vector(self, vector):
        if not self._opened and not self._oplot:
            self.set_w()
            self._opened = True

        if self._first:
            self._line = self.ax.plot(vector, '.-')
            self.fig.suptitle(self._title)
            self.ax.set_ylim(self._yrange)
            self._first = False
        else:
            self._line[0].set_ydata(vector)
        return self.fig
    
    def imshow(self, frame):
        if np.sum(self._wsize) == 0:
            size_frame = frame.shape
        else:
            size_frame = self._wsize

        if not self._opened:
            self.set_w(size_frame)
            self._opened = True
        if self._first:
            self.img = self.ax.imshow(frame)
            self._first = False
        else:
            self.img.set_data(frame)
            self.img.set_clim(frame.min(), frame.max())
        self.fig.canvas.draw()
        return self.fig
            

class DisplayServer():
    
    def __init__(self, params_dict, qin_, qout_):
        global qin, qout
        self.params_dict = params_dict
        self.t0 = time.time()
        qin = qin_
        qout = qout_
        
    def run(self):
        socketio.run(app, host='0.0.0.0', allow_unsafe_werkzeug=True)

    @socketio.on('newdata')
    def handle_newdata(args):
        '''Request for new data from the browser.
        1) Queue all requested object names
        2) Get back all data objects, plot them, and send them back to the browser
        '''
        # Queue plot requests
        for plot_name in args:
            qout.put(plot_name)
        qout.put(None)  # Terminator

        # Get all data objects and send their plots back to browser
        while True:
            name, obj_bytes = qin.get()
            if name is None: # Terminator
                break
            dataobj = pickle.loads(obj_bytes)
            print(dataobj)
            dataobj.xp = np  # Supply a numpy instance, sometimes it is needed
            if name not in plotters:
                plotters[name] = Plotter()
            fig = plotters[name].plot(dataobj)
            socketio.emit('plot', {'name': name, 'imgdata': encode(fig) })

        t1 = time.time()
        t0 = server.t0
        freq = 1.0 / (t1 - t0) if t1 != t0 else 0
        socketio.emit('done', f'Display rate: {freq:.2f} Hz')
        server.t0 = t1

    @socketio.on('connect')
    def handle_connect(*args):
        '''On connection, send the entire parameter dictionary
        so that the browser can refresh the view'''
        socketio.emit('params', server.params_dict)

    @app.route('/')
    def index():
        return render_template('specula_display.html')
        
    
def start_server(params_dict, qin, qout):
    global server
    server = DisplayServer(params_dict, qin, qout)
    server.run()

