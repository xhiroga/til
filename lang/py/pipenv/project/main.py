import requests

response = requests.get('https://httpbin.org/ip')

print(f"Your IP is {response.json()['origin']}")
