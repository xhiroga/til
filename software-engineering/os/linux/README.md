# Linux
Linuxの仕組み

# セットアップ
```console:
docker run -it ubuntu_test
```

# 用語
カーネル...カーネルモードで実行する(=通常プロセス(ユーザーモード)で実行すると困る)プログラム群
* プロセス管理システム
* メモリ管理システム
* デバイスドライバ

ファイルシステム...ストレージへのアクセスを担うプログラム

# 環境構築
4ステップからなる。  
1. パッケージ管理ソフトの更新  
2. 必要なパッケージの導入  
3. 実行可能ファイルを取得しなかった場合、ソースを取得&make
4. 不要なファイルの除去

```console:
apt-get update && apt-get upgrade -q -y && \
# パッケージの一覧を更新 & 導入済パッケージを更新

apt-get install -y --no-install-recommends golang-1.9 git make gcc libc-dev ca-certificates && \

git clone --depth 1 https://github.com/ethereum/go-ethereum && \
(cd go-ethereum && make geth) && \
cp go-ethereum/build/bin/geth /geth && \

apt-get remove -y golang-1.9 git make gcc libc-dev && apt autoremove -y && apt-get clean && \
# apt-get clean でアーカイブファイル(*.tar.gzとか)を削除
rm -rf /go-ethereum
```


# 参考
[［試して理解］Linuxのしくみ ～実験と図解で学ぶOSとハードウェアの基礎知識](https://www.amazon.co.jp/%EF%BC%BB%E8%A9%A6%E3%81%97%E3%81%A6%E7%90%86%E8%A7%A3%EF%BC%BDLinux%E3%81%AE%E3%81%97%E3%81%8F%E3%81%BF-%EF%BD%9E%E5%AE%9F%E9%A8%93%E3%81%A8%E5%9B%B3%E8%A7%A3%E3%81%A7%E5%AD%A6%E3%81%B6OS%E3%81%A8%E3%83%8F%E3%83%BC%E3%83%89%E3%82%A6%E3%82%A7%E3%82%A2%E3%81%AE%E5%9F%BA%E7%A4%8E%E7%9F%A5%E8%AD%98-%E6%AD%A6%E5%86%85-%E8%A6%9A-ebook/dp/B079YJS1J1)
[go-ethereum/containers/docker/develop-ubuntu/Dockerfile](https://github.com/ethereum/go-ethereum/blob/master/containers/docker/develop-ubuntu/Dockerfile)