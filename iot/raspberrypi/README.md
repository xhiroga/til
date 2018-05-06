# RaspberryPi


## Usage
## Setup
```
diskutil eraseDisk FAT32 RPI /dev/disk2 
# /dev/disk2のボリュームをFATA32形式でフォーマットし、パーティションにRPIと命名する。

diskutil unmountDisk /dev/disk2
# アンマウント

sudo dd bs=32m if=2018-04-18-raspbian-stretch.img of=/dev/disk2
# ddコマンドは標準入力→標準出力に決められたブロックサイズで流し込むものだが、それぞれファイルを指定することも可能。  
# 手元の環境だと4.9GBあたり2300秒かかった。
```
その他、パスワード変更、ロケール変更、キーボードレイアウトの変更など。

### 参考
フォーマット前
```
/dev/disk2 (internal, physical):
   #:                       TYPE NAME                    SIZE       IDENTIFIER
   0:     FDisk_partition_scheme                        *16.0 GB    disk2
   1:             Windows_FAT_32 NO NAME                 16.0 GB    disk2s1
```
フォーマット後(`diskutil eraseDisk FAT32 RPI /dev/disk2`)
```
/dev/disk2 (internal, physical):
   #:                       TYPE NAME                    SIZE       IDENTIFIER
   0:      GUID_partition_scheme                        *16.0 GB    disk2
   1:                        EFI EFI                     209.7 MB   disk2s1
   2:       Microsoft Basic Data RPI                     15.8 GB    disk2s2
```
dd後
```
/dev/disk2 (internal, physical):
   #:                       TYPE NAME                    SIZE       IDENTIFIER
   0:     FDisk_partition_scheme                        *16.0 GB    disk2
   1:             Windows_FAT_32 boot                    45.2 MB    disk2s1
   2:                      Linux                         4.9 GB     disk2s2
```

→ パーティションの数と種類が変わっているのがわかる。

# SSH接続
```console
# デフォルトではSSH無効のため有効化すること
ifconfig # RaspberryPi側/ wlan0のinetが接続先host
ssh pi@[host] # 作業端末側
```