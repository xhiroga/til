import http.server
import json
import threading

import zmq

RENDEZVOUS_PORT = 5555


class MainServiceHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            response = {
                "message": "メインサービスからの応答: リクエストを受け取りました"
            }
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")


def run_http_server(port=0):
    server = http.server.HTTPServer(("", port), MainServiceHandler)
    http_port = server.server_address[1]
    print(f"[*] HTTPサーバーがポート {http_port} で待機中...")

    # サーバーを別スレッドで起動
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()

    return http_port, server


def main():
    context = zmq.Context()

    rendezvous_socket = context.socket(zmq.REP)
    rendezvous_address = f"tcp://*:{RENDEZVOUS_PORT}"
    rendezvous_socket.bind(rendezvous_address)
    print(f"[*] ランデブーサービスが {rendezvous_address} で待機中...")

    http_port, http_server = run_http_server()
    main_service_address = f"http://localhost:{http_port}"
    print(f"[*] メインサービス（HTTPサーバー）のアドレス: {main_service_address}")

    print("[*] サーバー起動完了。Ctrl+Cで停止します。")

    try:
        while True:
            message = rendezvous_socket.recv_string()
            print(f"[Rendezvous] 受信: {message}")
            rendezvous_socket.send_string(main_service_address)
            print(f"[Rendezvous] 応答: {main_service_address}")

    except KeyboardInterrupt:
        print("\n[*] サーバーをシャットダウンします...")
    finally:
        http_server.shutdown()
        rendezvous_socket.close()
        context.term()
        print("[*] サーバー停止完了。")


if __name__ == "__main__":
    main()
