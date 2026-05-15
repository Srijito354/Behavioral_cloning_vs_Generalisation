import os


OUTPUT_FILE = "outputs/latest_report.txt"


def save_report(report):

    os.makedirs("outputs", exist_ok=True)

    with open(OUTPUT_FILE, "w") as f:
        f.write(report)