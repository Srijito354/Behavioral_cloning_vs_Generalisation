from transformers import AutoProcessor
from transformers import AutoModelForImageTextToText
from PIL import Image
import torch
import numpy as np

MODEL_NAME = "HuggingFaceTB/SmolVLM-Instruct-256M"

processor = AutoProcessor.from_pretrained(MODEL_NAME)


model = AutoModelForImageTextToText.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32
)

model.eval()


def extract_keyframes(clip):

    if len(clip) < 4:
        return [x["frame"] for x in clip]

    indices = [
        0,
        len(clip) // 3,
        (2 * len(clip)) // 3,
        len(clip) - 1
    ]

    return [clip[i]["frame"] for i in indices]


PROMPT = """
You are analyzing a behavioral cloning policy failure
in an autonomous driving simulator.

Tasks:
1. Identify the earliest visible indication of instability.
2. Analyze lane positioning drift.
3. Determine whether steering correction was delayed.
4. Explain why the trajectory became unrecoverable.
5. Suggest a corrective driving strategy.

Respond concisely and technically.
"""


def analyze_clip(clip):

    keyframes = extract_keyframes(clip)

    pil_images = [Image.fromarray(frame) for frame in keyframes]

    messages = [
        {
            "role": "user",
            "content": [
                *[
                    {
                        "type": "image",
                        "image": img
                    }
                    for img in pil_images
                ],
                {
                    "type": "text",
                    "text": PROMPT
                }
            ]
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=120
        )

    output = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    return output