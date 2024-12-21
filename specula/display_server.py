
import queue
import pickle
import multiprocessing as mp

from specula import cpuArray
from specula.base_processing_obj import BaseProcessingObj
from specula.base_value import BaseValue
from specula.data_objects.intensity import Intensity
from specula.data_objects.pixels import Pixels
from specula.data_objects.slopes import Slopes
from specula.data_objects.ef import ElectricField
from specula.display.data_plotter import DataPlotter

class ProcessingDisplay(BaseProcessingObj):
    '''
    Forwards data objects to a separate process using multiprocessing queues.
    '''
    def __init__(self, params_dict, input_ref_getter, output_ref_getter):
        super().__init__()
        self.qin = mp.Queue()
        self.qout = mp.Queue()
        
        p = mp.Process(target=start_server, args=(params_dict, self.qout, self.qin))  # Reversed queue order
        p.start()

        # Simulation speed calculation
        self.counter = 0
        self.t0 = time.time()
        self.c0 = self.counter
        self.speed_report = ''

        # Heuristic to detect inputs: they usually start with "in_"
        def data_obj_getter(name):
            if '.in_' in name:
                return input_ref_getter(name, target_device_idx=-1)
            else:
                try:
                    return output_ref_getter(name)     
                except ValueError:
                    # Try inputs as well
                    return input_ref_getter(name, target_device_idx=-1)

        self.data_obj_getter = data_obj_getter

    def trigger(self):
        t1 = time.time()
        self.counter += 1
        if t1 - self.t0 >= 1:
            niters = self.counter - self.c0
            self.speed_report = f"Simulation speed: {niters / (t1-self.t0):.2f} Hz"
            self.c0 = self.counter
            self.t0 = t1

        # Loop over data object requests
        # This loop is guaranteed to find an empty queue sooner or later,
        # thanks to the handshaking with the browser code, that will
        # avoid sending new requests until the None terminator is received
        # by the browser itself.

        while True:
            try:
                request = self.qin.get(block=False)
            except queue.Empty:
                return

            if request is None:
                self.qout.put((None, self.speed_report))
                return

            # Find the requested object, make sure it's on CPU,
            # and remove xp/np modules to prepare for pickling
            dataobj = self.data_obj_getter(request)
            if isinstance(dataobj, list):
                dataobj_cpu = [x.copyTo(-1) for x in dataobj]
            else:
                dataobj_cpu = dataobj.copyTo(-1)

            deleted = remove_xp_np(dataobj_cpu)

            # Double pickle trick (we pickle, and then qout will pickle again)
            # to avoid some problems
            # with serialization of modules, which apparently
            # are still present even after the xp and np removal

            obj_bytes = pickle.dumps(dataobj_cpu)
            self.qout.put((request, obj_bytes))

            # Put xp and np back in case this was a reference
            # and not a real copy
            putback_xp_np((dataobj_cpu, deleted))


from flask import Flask, render_template
from flask_socketio import SocketIO

import numpy as np
import io
import time
import base64


def encode(fig):
    '''
    Encode a PNG image for web display
    '''
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    imgB64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return imgB64


# Global variables needed by Flask-SocketIO            
app = Flask('Specula_display_server')
socketio = SocketIO(app)
server = None

class DisplayServer():
    '''
    Flask-SocketIO web server
    '''
    def __init__(self, params_dict, qin, qout):
        self.params_dict = params_dict
        self.t0 = time.time()
        self.qin = qin
        self.qout = qout
        self.plotters = {}
        
    def run(self):
        socketio.run(app, host='0.0.0.0', allow_unsafe_werkzeug=True)

    @socketio.on('newdata')
    def handle_newdata(args):
        '''Request for new data from the browser.
        1) Queue all requested object names
        2) Get back all data objects, plot them, and send them back to the browser
        '''
        print(args)
        # Queue plot requests
        for plot_name in args:
            server.qout.put(plot_name)
        server.qout.put(None)  # Terminator

        # Get all data objects and send their plots back to browser
        while True:
            name, obj_bytes = server.qin.get()
            if name is None: # Terminator
                speed_report = obj_bytes
                socketio.emit('speed_report', speed_report)
                break

            if name not in server.plotters:
                server.plotters[name] = DataPlotter()

            dataobj = pickle.loads(obj_bytes)
            if isinstance(dataobj, list):
                for obj in dataobj:
                    obj.xp = np
                fig = server.plotters[name].multi_plot(dataobj)
            else:
                dataobj.xp = np  # Supply a numpy instance, sometimes it is needed
                fig = server.plotters[name].multi_plot([dataobj])

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
        
        # Exclude DataStore since its input_list has a different format
        # and cannot be displayed at the moment
        display_params = {}
        for k, v in server.params_dict.items():
            if 'class' in v:
                if v['class'] == 'DataStore':
                    continue
            display_params[k] = v
        socketio.emit('params', display_params)

    @app.route('/')
    def index():
        return render_template('specula_display.html')
        
    
def start_server(params_dict, qin, qout):
    global server
    server = DisplayServer(params_dict, qin, qout)
    server.run()


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


def putback_xp_np(args):
    '''Put back the removed modules, if any.

    Works recursively on object lists
    '''
    obj, deleted = args
    if isinstance(obj, list):
        _ = list(map(putback_xp_np, zip(obj, deleted)))
        return

    for k, v in deleted.items():
        setattr(obj, k, v)