# bunの通信先を検証する

## 仮説

bunがパッケージをインストールする速度が早すぎる。独自のCDNを持っているのではないか？

## 検証

### 案1: dtrussを使用

```shell
# macos
sudo dtruss -n bun
# Another terminal
bun add lodash --no-cache --verbose
```

結果: ローカルのファイルへのアクセスは見えたものの、ネットワークI/Oはよく分からなかった。

### 案2: tcpdumpを使用

```shell
sudo tcpdump -k NP > log.pcap
# Another terminal
bun add lodash --no-cache --verbose
# Ctrl+C to stop tcpdump
cat log.pcap | grep bun
```

結果: 通信しているIPアドレスは分かったが、それだけでは不十分。
