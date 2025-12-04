import cv2
import json
import urllib.request
import numpy as np
import websocket #NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
from contextlib import contextmanager
from requests_toolbelt import MultipartEncoder


@contextmanager
def comfy_ws(address: str, client_id: str):
    """
    Контекст-менеджер: открываем WS к ComfyUI,
    возвращаем объект websocket.WebSocket,
    закрываем даже при исключениях.
    """
    url = f"ws://{address}/ws?clientId={client_id}"
    ws = websocket.create_connection(url)

    try:
        yield ws
    finally:
        ws.close()

def upload_image(server_address, input_path, name, image_type="input", overwrite=False):
    with open(input_path, 'rb') as file:
        multipart_data = MultipartEncoder(
            fields={
                'image': (name, file, 'image/png'),
                'type': image_type,
                'overwrite': str(overwrite).lower()
            }
        )

        headers = {'Content-Type': multipart_data.content_type}

        request = urllib.request.Request(f"http://{server_address}/upload/image", data=multipart_data, headers=headers)
        with urllib.request.urlopen(request) as response:
            return response.read()

def queue_prompt(prompt, server_address, client_id):
    p = {"prompt": prompt, "client_id": str(client_id)}
    data = json.dumps(p).encode('utf-8')
    req =  urllib.request.Request("http://{}/prompt".format(server_address), data=data)
    return json.loads(urllib.request.urlopen(req).read())

def get_images(prompt, ws, client_id, server_address):
    prompt_id = queue_prompt(prompt, server_address, client_id)['prompt_id']
    output_images = {}
    current_node = ""
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['prompt_id'] == prompt_id:
                    if data['node'] is None:
                        break
                    else:
                        current_node = data['node']
        else:
            if current_node == 'save_image_websocket_node':
                images_output = output_images.get(current_node, [])
                images_output.append(out[8:])
                output_images[current_node] = images_output

    return output_images