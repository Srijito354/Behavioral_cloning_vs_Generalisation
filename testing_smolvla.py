import torch
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from custom_dataset import Data, DataLoader
from torchvision.transforms import transforms
from transformers.image_utils import load_image
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_8bit=True)
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
model = AutoModelForImageTextToText.from_pretrained(
    "HuggingFaceTB/SmolVLM-256M-Instruct",
    quantization_config=quantization_config,
    #torch_dtype=torch.bfloat8,
)

df = pd.read_csv("driving_log.csv")
mask = df["Steering"] < -0.8
masked_df = df[mask]

for batch in masked_df["Center"]:

    print(batch)
    image = Image.open(batch.removeprefix("/media/dev-mk2/Expansion/driving_data/"))

    messages = [
        {
            "role": "user",
            "content": [
                #{"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
                {"type": "image", "image": image},
                #{"type": "text", "text": "What animal is on the candy?"}
                #{"type": "text", "text": "Describe the image."}
                {"type": "text", "text": "What should a car do on this road? (and explain, why)"}
            ]
        },
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100)

    print("\n\n")
    print(processor.decode(outputs[0][inputs["input_ids"].shape[-1]:]))

    plt.imshow(np.array(image))
    plt.show()

    break