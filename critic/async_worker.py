import threading

from critic.vlm_critic import analyze_clip
from critic.report_generator import save_report


class CriticWorker:
    def __init__(self):
        self.running = False

    def launch(self, clip):

        if self.running:
            return

        thread = threading.Thread(
            target=self.run,
            args=(clip,)
        )

        thread.start()

    def run(self, clip):

        self.running = True

        try:
            report = analyze_clip(clip)
            print("=== VLM FAILURE ANALYSIS ===")
            print(report)
            save_report(report)

        except Exception as e:
            print("Critic error:", e)

        self.running = False