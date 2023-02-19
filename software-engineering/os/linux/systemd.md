# systemd

# Usage
/etc/systemd/system以下に、 `.service`ファイルを作成する(Unitと呼ばれる)  
`systemctl start *.service`でチェックしたのち、`systemctl enable *.service`でdaemonにserviceを登録する

# MEMO
* ExecStartやWorkingDirectoryなどの引数は全て絶対パスである必要がある

# TIPS
* プロセスが動かない時は、`status`だけではなく/var/log/syslogも見ること
* Pythonのように普段はソースとして扱っているものでも、ブートストラップとして使うなら`chmod +x`が必要になる