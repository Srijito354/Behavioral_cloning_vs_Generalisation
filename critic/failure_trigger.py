class FailureTrigger:
    def __init__(self):
        self.low_speed_counter = 0

    def check(self, speed, steering):

        if speed < 0.8:
            self.low_speed_counter += 1
        else:
            self.low_speed_counter = 0

        excessive_steering = abs(steering) > 0.85

        stalled = self.low_speed_counter > 15

        return stalled or excessive_steering