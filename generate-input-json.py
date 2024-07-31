import json
import requests

# install required packages before running
# pip install pillow numpy --upgrade
from PIL import Image
import numpy as np

# method to generate payload from image url
def generate_payload(image_url):
    # download image from url and resize
    image_inputs = Image.open(requests.get(image_url, stream=True).raw)
    image_inputs = image_inputs.resize((200, 200))

    # convert image to numpy array
    image_tensor = np.asarray(image_inputs)
    # derive image shape
    image_shape = [1] + list(image_tensor.shape)

    # create payload request
    payload = {
        "id": "0",
        "inputs": [
            {
                "name": "input_tensor",
                "shape": image_shape,
                "datatype": "UINT8",
                "parameters": {},
                "data": image_tensor.tolist(),
            }
        ],
    }

    # save payload as json file
    triton_payload_file = "instances.json"
    with open(triton_payload_file, "w") as f:
        json.dump(payload, f)
    print(f"Triton payload json generated at {triton_payload_file}")

if __name__ == '__main__':
  image_url = "https://github.com/tensorflow/models/raw/master/research/object_detection/test_images/image2.jpg"
  generate_payload(image_url)