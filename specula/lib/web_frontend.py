'''
pip install flask-socketio python-socketio requests
'''
from flask import Flask, render_template, request
from flask_socketio import SocketIO

app = Flask('Specula_frontend')
socketio = SocketIO(app)

'''
From clients:

import socketio
sio_client = socketio.Client()
sio_client.connect('http://localhost:5000')
sio_client.emit('send_data',{"key": "value"})
'''

@socketio.on('simul_update')
def simul_update(data):
    '''
    Message received from running simulations with: (name, data, port)
    '''
    client_id = request.sid
    print(client_id, data)
    socketio.emit('simul_update', data)

@app.route('/specula')
def index():
    return render_template('specula_frontend.html')
  
if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=8080)