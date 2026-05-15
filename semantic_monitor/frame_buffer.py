from collections import deque
import time


class SemanticFrameBuffer:

    def __init__(self, max_frames=30):
        self.frames = deque(maxlen=max_frames)

    def add(self, frame):
        self.frames.append({
            "timestamp": time.time(),
            "frame": frame
        })

    def latest(self):
        return list(self.frames)