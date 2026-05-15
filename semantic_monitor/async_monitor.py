import threading

from semantic_monitor.semantic_monitor import (
    analyze_scene
)


class AsyncSemanticMonitor:

    def __init__(self):

        self.running = False

        self.latest_result = {
            "steering_delta": 0.0,
            "throttle_delta": 0.0,
            "raw_output": ""
        }

    def launch(
        self,
        frame,
        steering_history,
        throttle_history
    ):

        if self.running:
            return

        thread = threading.Thread(
            target=self.run,
            args=(
                frame,
                steering_history,
                throttle_history
            )
        )

        thread.start()

    def run(
        self,
        frame,
        steering_history,
        throttle_history
    ):

        self.running = True

        try:

            result = analyze_scene(
                frame,
                steering_history,
                throttle_history
            )

            self.latest_result = result

            print("\n=== SEMANTIC ADVISOR ===\n")

            print(result["raw_output"])

            with open(
                "outputs/latest_warning.txt",
                "w"
            ) as f:

                f.write(result["raw_output"])

        except Exception as e:

            print("Semantic monitor error:", e)

        self.running = False