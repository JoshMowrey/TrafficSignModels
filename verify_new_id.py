import requests

TOKEN = "MLY|25326284683690643|ada0b032bdad9e742f2a702909c47d92"
NEW_ID = "919316338641544"

url = f"https://graph.mapillary.com/{NEW_ID}"
headers = {"Authorization": f"OAuth {TOKEN}"}
params = {"fields": "id,geometry"}

print(f"Testing New ID: {url}")

try:
    response = requests.get(url, headers=headers, params=params)
    print(response.status_code)
    print(response.json())
except Exception as e:
    print(e)
