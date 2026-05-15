class ControlFusion:

    def __init__(self):

        self.semantic_weight = 0.15

    def fuse(
        self,
        bc_steering,
        bc_throttle,
        semantic_steering_delta,
        semantic_throttle_delta
    ):

        final_steering = (
            bc_steering
            +
            self.semantic_weight
            * semantic_steering_delta
        )

        final_throttle = (
            bc_throttle
            +
            self.semantic_weight
            * semantic_throttle_delta
        )

        final_steering = max(
            -1.0,
            min(1.0, final_steering)
        )

        final_throttle = max(
            0.0,
            min(1.0, final_throttle)
        )

        return (
            final_steering,
            final_throttle
        )