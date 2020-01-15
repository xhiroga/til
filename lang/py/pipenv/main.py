import requests

if __name__ == '__main__':
    response = requests.get('https://httpbin.org/ip')
    print(f"Your IP is {response.json()['origin']}")
