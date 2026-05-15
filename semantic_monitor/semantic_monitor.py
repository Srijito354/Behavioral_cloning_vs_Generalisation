from transformers import AutoProcessor
from transformers import AutoModelForImageTextToText

from PIL import Image

import torch
import re

from semantic_monitor.prompts import (
    SEMANTIC_PROMPT_TEMPLATE
)


MODEL_NAME = "HuggingFaceTB/SmolVLM-Instruct-256M"


processor = AutoProcessor.from_pretrained(
    MODEL_NAME
)

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32
)

model.eval()


def parse_delta(text, field_name, default=0.0):

    try:

        pattern = rf"{field_name}:\\s*([-+]?[0-9]*\\.?[0-9]+)"

        match = re.search(pattern, text)

        if match:
            return float(match.group(1))

    except:
        pass

    return default


def analyze_scene(
    frame,
    steering_history,
    throttle_history
):

    image = Image.fromarray(frame)

    prompt = SEMANTIC_PROMPT_TEMPLATE.format(
        steering_history=steering_history,
        throttle_history=throttle_history
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image
                },
                {
                    "type": "text",
                    "text": prompt
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

    steering_delta = parse_delta(
        output,
        "Steering Correction Delta",
        default=0.0
    )

    throttle_delta = parse_delta(
        output,
        "Throttle Correction Delta",
        default=0.0
    )

    steering_delta = max(
        -0.2,
        min(0.2, steering_delta)
    )

    throttle_delta = max(
        -0.3,
        min(0.0, throttle_delta)
    )

    return {
        "raw_output": output,
        "steering_delta": steering_delta,
        "throttle_delta": throttle_delta
    }