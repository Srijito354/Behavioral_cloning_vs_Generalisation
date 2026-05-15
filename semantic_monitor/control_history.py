from collections import deque


class ControlHistory:

    def __init__(self, max_history=20):

        self.steering = deque(maxlen=max_history)
        self.throttle = deque(maxlen=max_history)

    def add(self, steering, throttle):

        self.steering.append(round(float(steering), 4))
        self.throttle.append(round(float(throttle), 4))

    def get_history(self):

        return {
            "steering": list(self.steering),
            "throttle": list(self.throttle)
        }