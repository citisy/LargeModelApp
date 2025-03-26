import json
import uuid

import requests
import websocket

from workflows.skeletons import Module


class Model(Module):
    """base on apis of ComfyUI
    https://github.com/comfyanonymous/ComfyUI"""

    host = '127.0.0.1'
    port = 8188

    def request(self, prompt):
        client_id = str(uuid.uuid4())
        address = f'{self.host}:{self.port}'

        ws = websocket.WebSocket()
        ws.connect(f"ws://{address}/ws?clientId={client_id}")

        p = {"prompt": prompt, "client_id": client_id}
        req = requests.post(f"http://{address}/prompt", json=p)
        prompt_id = req.json()['prompt_id']

        while True:
            out = ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                if message['type'] == 'executing':
                    data = message['data']
                    if data['node'] is None and data['prompt_id'] == prompt_id:
                        break
            else:
                continue

        req = requests.get(f"http://{address}/history/{prompt_id}")
        history = req.json()[prompt_id]
        return history

    def request_images(self, prompt):
        address = f'{self.host}:{self.port}'
        history = self.request(prompt)
        output_images = {}
        for node_id in history['outputs']:
            node_output = history['outputs'][node_id]
            if 'images' in node_output:
                images_output = []
                for image in node_output['images']:
                    data = {"filename": image['filename'], "subfolder": image['subfolder'], "type": image['type']}
                    req = requests.get(f"http://{address}/view", params=data)
                    image['image_data'] = req.content
                    images_output.append(image)
                output_images[node_id] = images_output

        return output_images
