from collections import deque
import time


class ClipBuffer:
    def __init__(self, max_frames=100):
        self.frames = deque(maxlen=max_frames)

    def add(self, frame, telemetry):
        self.frames.append({
            "timestamp": time.time(),
            "frame": frame,
            "telemetry": telemetry
        })

    def get_clip(self):
        return list(self.frames)