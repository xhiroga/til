# IPアドレスとホスト名を/etc/hostsに追加する

```bash
# /etc/hostsにリダイレクトするためにはroot権限が必要だが、sudoでリダイレクトはできない。
# したがって、shにコマンド文字列を与えて解決する。
sudo sh -c 'echo "192.168.XXX.XXX raspberry.pi" >> /etc/hosts'
```