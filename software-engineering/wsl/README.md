# WSL

## mount

WSL2は9PプロトコルとWindows APIを経由することで、Windowsのファイルシステムにアクセスすることが可能である。

```console
$ ls /mnt/c
```

しかし、WSL2のネイティブなファイルシステムとは違い、速度に問題がある。

```
# WSL2のネイティブなファイルシステム
% dd if=/dev/zero of=/home/hiroga/test.img bs=1G count=1 oflag=dsync
1+0 records in
1+0 records out
1073741824 bytes (1.1 GB, 1.0 GiB) copied, 1.15242 s, 932 MB/s
% dd if=/home/hiroga/test.img of=/dev/null bs=1G
1+0 records in
1+0 records out
1073741824 bytes (1.1 GB, 1.0 GiB) copied, 0.521108 s, 2.1 GB/s
% rm /home/hiroga/test.img

% dd if=/dev/zero of=/mnt/c/Users/hiroga/test.img bs=1G count=1 oflag=dsync
flag=dsync
1+0 records in
1+0 records out
1073741824 bytes (1.1 GB, 1.0 GiB) copied, 4.19622 s, 256 MB/s
% dd if=/mnt/c/Users/hiroga/test.img of=/dev/null bs=1G
1+0 records in
1+0 records out
1073741824 bytes (1.1 GB, 1.0 GiB) copied, 3.96538 s, 271 MB/s
% rm /mnt/c/Users/hiroga/test.img
```
