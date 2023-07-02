import os

from atproto import Client
from dotenv import load_dotenv

load_dotenv(verbose=True)

HANDLE = os.getenv('HANDLE')
PASSWORD = os.getenv('PASSWORD')


def main():
    client = Client()
    profile = client.login(HANDLE, PASSWORD)
    print('Welcome,', profile.displayName)

    post_ref = client.send_post(text='Hello World from Python!')
    client.like(post_ref)


if __name__ == '__main__':
    main()
