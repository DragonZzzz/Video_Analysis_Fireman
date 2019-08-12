import requests

data = {
  "video": "./TestData/False2-fall.mp4",
  "ckpt": "./C3D/ckpt",
  "threshold": 0.8,
  "step": 30,
  "length": 15
}

res = requests.post('http://127.0.0.1:5000/detect', data=data)
print(res.text)