import sys

import requests
import zmq

RENDEZVOUS_PORT = 5555


def main():
    context = zmq.Context()

    print("[*] クライアント起動")

    rendezvous_socket = context.socket(zmq.REQ)
    rendezvous_address = f"tcp://localhost:{RENDEZVOUS_PORT}"
    print(f"[*] ランデブーサービス ({rendezvous_address}) に接続中...")
    rendezvous_socket.connect(rendezvous_address)

    print("[*] メインサービスのアドレスを問い合わせます...")
    rendezvous_socket.send_string("discover_main_service")

    try:
        if rendezvous_socket.poll(5000, zmq.POLLIN):
            main_service_address = rendezvous_socket.recv_string()
            print(f"[*] 受信したメインサービスアドレス: {main_service_address}")
        else:
            print("[エラー] ランデブーサービスからの応答がタイムアウトしました。")
            rendezvous_socket.close()
            context.term()
            sys.exit(1)

    except zmq.ZMQError as e:
        print(f"[エラー] ランデブーサービスとの通信中にエラーが発生しました: {e}")
        rendezvous_socket.close()
        context.term()
        sys.exit(1)
    finally:
        rendezvous_socket.close()

    print(f"[*] メインサービス ({main_service_address}) に接続中...")

    print("[*] メインサービスへGETリクエスト送信")
    try:
        response = requests.get(main_service_address, timeout=5)
        if response.status_code == 200:
            reply = response.json()
            print(f"[*] メインサービスから受信: {reply}")
        else:
            print(f"[エラー] メインサービスからの応答がエラー: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"[エラー] メインサービスとの通信中にエラーが発生しました: {e}")

    print("[*] クライアント処理完了")

    context.term()


if __name__ == "__main__":
    main()
