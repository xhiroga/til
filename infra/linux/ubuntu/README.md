# Ubuntu

誰にでも使いやすいことをミッションとしたLinuxディストリビューション。Debianの派生。  
デスクトップ環境が充実している。  

# 実行方法
```Console
docker run -i ubuntu
```

# ツール
## パッケージ管理
内部的にAdvanced Package Toolを用いており、インターフェースとしてはapt-get, apt, aptitudeがある。  
```Console
apt update # 環境構築直後は最新のパッケージが入っていないので、installがunable to locate扱いされてしまうかも。
apt install # -yオプションをつけると問い合わせに全てイエスで答える。Dockerfile作成時などに使用。
apt-get update && apt-get install -y software-properties-common && add-apt-repository -y ppa:ethereum/ethereum
# PPA(パーソナル・パッケージ・マネージャー)の提供するパッケージをインストールする。
# インストール後はアップデートが必要。
```
