class InterventionController:

    def __init__(self):
        self.risk_state = "LOW"

    def parse_risk(self, report):

        if "HIGH" in report:
            return "HIGH"

        if "MEDIUM" in report:
            return "MEDIUM"

        return "LOW"

    def intervene(self, throttle, steering, risk):

        if risk == "HIGH":
            throttle *= 0.4

        elif risk == "MEDIUM":
            throttle *= 0.7

        return throttle, steering