from gevent import monkey
monkey.patch_all()  # MUST be first

import cv2
import numpy as np
import socketio
import base64
import torch
from PIL import Image
from io import BytesIO
from flask import Flask
from torchvision import transforms
from Model import Modela
import gevent.pywsgi
from geventwebsocket.handler import WebSocketHandler

sio = socketio.Server(async_mode='gevent', logger=True, engineio_logger=True)
app = socketio.WSGIApp(sio, Flask(__name__))

speed_limit = 15

model = Modela()
#model.load_state_dict(torch.load("trained_model_pre(non-panorama-only-center).pt"))
model.load_state_dict(torch.load("fine_tuned_resnet_400.pt"))
model.eval()

transform = transforms.Compose((
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
))

def send_control(steering, throttle):
    print("Sending!")
    sio.emit('steer', data={
        'steering_angle': str(steering),
        'throttle':       str(throttle),
        #'brake': str(brake)
    })

@sio.on('connect')
def connect(sid, environ):
    print("Simulator connected!", sid)
    send_control(0, 0)

@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        speed = float(data['speed'])
        img = Image.open(BytesIO(base64.b64decode(data['image']))).convert("RGB")

        img_tensor = transform(img).unsqueeze(0)

        print(img_tensor.shape)

        with torch.no_grad():
            out = model(img_tensor)

        steering = float(torch.tanh(out[:, 0]).item())
        throttle = 1.0 - speed / speed_limit
        '''
        if (speed < speed_limit):
            throttle = throttle/2
        else:
            throttle = 0.0
        '''
        brake = float(out[:, 2].item())

        print(f"Steering: {steering:.4f} | Throttle: {throttle:.4f} | Speed: {speed:.2f}")
        send_control(steering, throttle)

if __name__ == '__main__':
    print("Listening on port 4567...")
    server = gevent.pywsgi.WSGIServer(
        ('', 4567),
        app,
        handler_class=WebSocketHandler
    )
    server.serve_forever()
