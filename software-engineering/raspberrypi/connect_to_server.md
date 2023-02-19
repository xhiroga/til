# connect to server
Raspberry PiをMacbookから操作するためのテクニック集。(IPアドレスは例)  

## IP Addressを調べる
Raspberry Piを接続する前後でアドレス変換テーブルのエントリーを確認すればよい。  
`arp -a`(-aでhostnameを指定せずエントリーを表示)

## AFP(Apple Filing Protocol)
Finder > Go > Connect to Server
`afp://192.168.XXX.XXX`

## VNC(Virtual Network Computing)
Finder > Go > Connect to Server
`vnc://192.168.XXX.XXX:5901`

