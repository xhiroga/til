# 公開鍵認証
パスワードではなく、公開鍵/秘密鍵の組み合わせで認証を行う方式。  

```shell
# -t ed25519でEdDSAを利用できる。鍵の名前が重複しないように-fで命名する。
ssh-keygen -t ed25519 -a 100 -f ~/.ssh/id_ed25519_raspberry
# ssh接続するホスト名にエイリアスを付与したり、秘密鍵をあらかじめ設定しておく。
vi ~/.ssh/config
# sshサーバーの~/.sshに公開鍵を置く(そうしない場合、普通に今まで通りパスワードを要求される)
scp ~/.ssh/id_ed25519_raspberry.pub pi@raspberry:~/.ssh/authorized_keys
```