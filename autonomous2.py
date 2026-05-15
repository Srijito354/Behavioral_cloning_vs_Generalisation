from gevent import monkey
monkey.patch_all()

import socketio
import base64
import torch
import numpy as np

from PIL import Image
from io import BytesIO
from flask import Flask
from torchvision import transforms

import gevent.pywsgi
from geventwebsocket.handler import WebSocketHandler

from Model import Modela

from critic.clip_buffer import ClipBuffer
from critic.failure_trigger import FailureTrigger
from critic.async_worker import CriticWorker


sio = socketio.Server(async_mode='gevent')
app = socketio.WSGIApp(sio, Flask(__name__))

speed_limit = 15

model = Modela()
model.load_state_dict(torch.load("fine_tuned_resnet_400.pt"))
model.eval()


buffer = ClipBuffer(max_frames=100)
trigger = FailureTrigger()
critic = CriticWorker()


transform = transforms.Compose((
    transforms.Lambda(
        lambda img: img.crop((0, 60, img.width, img.height - 25))
    ),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
))

def send_control(steering, throttle):

    sio.emit('steer', data={
        'steering_angle': str(steering),
        'throttle': str(throttle)
    })


@sio.on('connect')
def connect(sid, environ):
    print("Simulator connected!", sid)
    send_control(0, 0)


@sio.on('telemetry')
def telemetry(sid, data):

    if not data:
        return

    speed = float(data['speed'])

    img = Image.open(
        BytesIO(base64.b64decode(data['image']))
    ).convert("RGB")

    raw_frame = np.array(img)

    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        out = model(img_tensor)

    steering = float(torch.tanh(out[:, 0]).item())

    throttle = 1.0 - speed / speed_limit

    telemetry_data = {
        "speed": speed,
        "steering": steering,
        "throttle": throttle
    }

    buffer.add(raw_frame, telemetry_data)

    failed = trigger.check(speed, steering)

    if failed:

        print("Failure detected. Launching critic...")

        clip = buffer.get_clip()

        critic.launch(clip)

    print(
        f"Steering: {steering:.4f} | "
        f"Throttle: {throttle:.4f} | "
        f"Speed: {speed:.2f}"
    )

    send_control(steering, throttle)


if __name__ == '__main__':

    print("Listening on port 4567...")

    server = gevent.pywsgi.WSGIServer(
        ('', 4567),
        app,
        handler_class=WebSocketHandler
    )

    server.serve_forever()