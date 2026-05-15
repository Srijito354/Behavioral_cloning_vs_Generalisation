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
from geventwebsocket.handler import (
    WebSocketHandler
)

from Model import Modela

from semantic_monitor.async_monitor import (
    AsyncSemanticMonitor
)

from semantic_monitor.control_history import (
    ControlHistory
)

from semantic_monitor.control_fusion import (
    ControlFusion
)


sio = socketio.Server(async_mode='gevent')

app = socketio.WSGIApp(
    sio,
    Flask(__name__)
)

speed_limit = 15


model = Modela()

model.load_state_dict(
    torch.load("fine_tuned_resnet_400.pt")
)

model.eval()


semantic_monitor = AsyncSemanticMonitor()

control_history = ControlHistory()

fusion = ControlFusion()


transform = transforms.Compose((

    transforms.Lambda(
        lambda img: img.crop(
            (0, 60, img.width, img.height - 25)
        )
    ),

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
))


semantic_counter = 0


def send_control(steering, throttle):

    sio.emit(
        'steer',
        data={
            'steering_angle': str(steering),
            'throttle': str(throttle)
        }
    )


@sio.on('connect')
def connect(sid, environ):

    print("Simulator connected!")

    send_control(0, 0)


@sio.on('telemetry')
def telemetry(sid, data):

    global semantic_counter

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

    bc_steering = float(
        torch.tanh(out[:, 0]).item()
    )

    bc_throttle = 1.0 - speed / speed_limit


    control_history.add(
        bc_steering,
        bc_throttle
    )


    semantic_counter += 1

    if semantic_counter % 20 == 0:

        history = control_history.get_history()

        semantic_monitor.launch(
            raw_frame,
            history["steering"],
            history["throttle"]
        )


    semantic_result = (
        semantic_monitor.latest_result
    )

    final_steering, final_throttle = (
        fusion.fuse(
            bc_steering,
            bc_throttle,
            semantic_result["steering_delta"],
            semantic_result["throttle_delta"]
        )
    )


    print(
        f"BC Steering: {bc_steering:.4f} | "
        f"Semantic Delta: "
        f"{semantic_result['steering_delta']:.4f} | "
        f"Final Steering: {final_steering:.4f}"
    )

    print(
        f"BC Throttle: {bc_throttle:.4f} | "
        f"Semantic Delta: "
        f"{semantic_result['throttle_delta']:.4f} | "
        f"Final Throttle: {final_throttle:.4f}"
    )

    send_control(
        final_steering,
        final_throttle
    )


if __name__ == '__main__':

    print("Listening on port 4567...")

    server = gevent.pywsgi.WSGIServer(
        ('', 4567),
        app,
        handler_class=WebSocketHandler
    )

    server.serve_forever()