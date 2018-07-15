# Tools
ネットワーク系の調査で使用するツール群。

## ping
特定のサーバーがresponseするのではなく、OSが直接responseする。  

## ip
ネットワークインターフェースに割り当てられているアドレスを表示する。  
ネットワークインターフェースの起動・停止もできる。設定変更後の再起動の代わりに落とし上げるなど。  

## netcat-openbsd
TCP/UDPの読み書きのための万能ツール。  
```
# TCPクライアント
nc localhost 3456

# TCPサーバー
nc -l 3456

# HTTPリクエスト(接続先ポートのアプリケーション動作確認)
echo "GET / HTTP/1.1\n\n" | nc example.com 80

# HTTPサーバー
while : ; do (echo -ne "HTTP/1.0 200 Ok\nContent-Length: $(wc -c < response.txt)\n\n"; cat pwdresponse.txt) | nc -l -p 80; done

# ポートスキャン
nc -vz localhost 1-65535 2>&1 | grep succeeded
```

## tcpdump
基本的には対象のホストやポート/プロトコルを絞り込んでキャプチャする。
```bash
tcpdump host google.com
tcpdump tcp # httpのようにアプリケーションレイヤーのプロトコルは指定できないようだ。
tcpdump port 80
tcpdump host yaoo.co.jp and port 80
```

## traceroute
目的のhostにパケットを送るまでに通ったrouteをprintする。  
対象のホストまでのどこで通信障害が起きているのかを知ることができる。  
`traceroute google.com`

## mtr
My tracerouteの略のよう。ping + tracerouteのようなアプリケーション。  
`mtr google.com`


[原因調査用Linuxコマンド - 外道父の道](http://blog.father.gedow.net/2012/10/23/linux-command-for-trouble/)