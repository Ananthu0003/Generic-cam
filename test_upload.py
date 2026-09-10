import requests

url = 'http://127.0.0.1:8000/api/model/upload'
with open('sample_bracket.step', 'rb') as f:
    files = {'file': ('sample_bracket.step', f, 'application/octet-stream')}
    resp = requests.post(url, files=files)
    print(f'Status: {resp.status_code}')
    if resp.status_code == 200:
        data = resp.json()
        print(f'Model name: {data["name"]}')
        print(f'Triangles: {data["mesh"]["triangle_count"]}')
        print(f'Faces: {data["mesh"]["face_count"]}')
        print(f'BBox min: {data["mesh"]["bounding_box"]["min"]}')
        print(f'BBox max: {data["mesh"]["bounding_box"]["max"]}')
    else:
        print(f'Error: {resp.text}')
