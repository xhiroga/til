# sudoers
sudoの権限を管理するためのファイル(/etc/sudoers)。  

# Memo
`%sudo ALL=(ALL) NOPASSWD:ALL`  
* ALL=(ALL): ALLホストではALLユーザーに成り変われる。  
* NOPASSWD:ALL: %sudoグループのALLコマンドにパスワードなしでのsudoを実行可能にする。  
