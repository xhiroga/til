# docker

## volume

### Backup & Restore

設定は[Homelab](https://github.com/xhiroga/homelab/blob/main/apps/n8n/docker-compose.yml)を参照。

Volumeのバックアップにおいてパーミッション・所有者などのメタデータを保ちたい。[Ciffelia氏のZennの記事](https://zenn.dev/ciffelia/articles/docker-volume-backup-restore)も参照。

しかし、少なくともWSL2の場合は、ホスト側でバックアップ・リストアすると、コンテナ側のパーミッションが失われる。

```console
$ docker compose -f n8n/docker-compose.yml -p n8n-borzoi --env-file config/.n8n-borzoi.env run --rm -it --entrypoint /bin/sh n8n
~ $ cd .n8n/
~/.n8n $ ls -la
total 472
drwxr-sr-x    5 node     node          4096 Jun  3 01:29 .
drwxr-sr-x    1 node     node          4096 Jun  3 01:29 ..
drwxr-sr-x    2 node     node          4096 Jun  3 01:29 binaryData
-rw-r--r--    1 node     node            56 Jun  3 01:29 config
-rw-r--r--    1 node     node             0 Jun  3 01:29 crash.journal
-rw-r--r--    1 node     node        454656 Jun  3 01:29 database.sqlite
drwxr-sr-x    2 node     node          4096 Jun  3 01:29 git
-rw-r--r--    1 node     node             0 Jun  3 01:29 n8nEventLog.log
drwxr-sr-x    2 node     node          4096 Jun  3 01:29 ssh

$ sudo mount -t drvfs '\\wsl.localhost\docker-desktop\mnt\docker-desktop-disk\data\docker' /var/lib/docker
$ VOLUME_NAME="n8n-borzoi_n8n_data"
$ mkdir -p .local
$ tar -czf .local/$VOLUME_NAME.tar.gz -C /var/lib/docker/volumes/$VOLUME_NAME _data
$ tar -tvf .local/$VOLUME_NAME.tar.gz
drwxrwxrwx root/root         0 2025-06-03 10:10 _data/
-rwxrwxrwx root/root         0 2025-06-03 10:10 _data/crash.journal
drwxrwxrwx root/root         0 2025-06-03 10:10 _data/binaryData/
-rwxrwxrwx root/root         0 2025-06-03 10:10 _data/n8nEventLog.log
drwxrwxrwx root/root         0 2025-06-03 10:10 _data/git/
-rwxrwxrwx root/root        56 2025-06-03 10:10 _data/config
drwxrwxrwx root/root         0 2025-06-03 10:10 _data/ssh/
-rwxrwxrwx root/root    454656 2025-06-03 10:10 _data/database.sqlite

$ docker volume rm "$VOLUME_NAME"
$ docker volume create "$VOLUME_NAME"
$ sudo rm -r "/var/lib/docker/volumes/$VOLUME_NAME/_data"
$ sudo tar -xzf .local/$VOLUME_NAME.tar.gz -C "/var/lib/docker/volumes/$VOLUME_NAME" --preserve-permissions --numeric-owner
$ ls -la /var/lib/docker/volumes/$VOLUME_NAME/_data
total 448
drwxrwxrwx 5 root root    512 Jun  3 10:10 .
drwxrwxrwx 3 root root    512 Jun  3 10:11 ..
drwxrwxrwx 2 root root    512 Jun  3 10:10 binaryData
-rwxrwxrwx 1 root root     56 Jun  3 10:10 config
-rwxrwxrwx 1 root root      0 Jun  3 10:10 crash.journal
-rwxrwxrwx 1 root root 454656 Jun  3 10:10 database.sqlite
drwxrwxrwx 2 root root    512 Jun  3 10:10 git
-rwxrwxrwx 1 root root      0 Jun  3 10:10 n8nEventLog.log
drwxrwxrwx 2 root root    512 Jun  3 10:10 ssh

$ cd /home/hiroga/Documents/GitHub/homelab/apps
$ docker compose -f n8n/docker-compose.yml -p n8n-borzoi --env-file config/.n8n-borzoi.env up -d
$ docker compose -f n8n/docker-compose.yml -p n8n-borzoi --env-file config/.n8n-borzoi.env logs -f
n8n-1   | Permissions 0644 for n8n settings file /home/node/.n8n/config are too wide. This is ignored for now, but in the future n8n will attempt to change the permissions automatically. To automatically enforce correct permissions now set N8N_ENFORCE_SETTINGS_FILE_PERMISSIONS=true (recommended), or turn this check off set N8N_ENFORCE_SETTINGS_FILE_PERMISSIONS=false.
n8n-1   | User settings loaded from: /home/node/.n8n/config
n8n-1   | Last session crashed
n8n-1   | Error: EACCES: permission denied, open '/home/node/.n8n/crash.journal'
n8n-1   |     at open (node:internal/fs/promises:639:25)
n8n-1   |     at touchFile (/usr/local/lib/node_modules/n8n/dist/crash-journal.js:18:20)
n8n-1   |     at Object.init (/usr/local/lib/node_modules/n8n/dist/crash-journal.js:32:5)
n8n-1   |     at Start.initCrashJournal (/usr/local/lib/node_modules/n8n/dist/commands/base-command.js:135:9)
n8n-1   |     at Start.init (/usr/local/lib/node_modules/n8n/dist/commands/start.js:141:9)
n8n-1   |     at Start._run (/usr/local/lib/node_modules/n8n/node_modules/@oclif/core/lib/command.js:301:13)
n8n-1   |     at Config.runCommand (/usr/local/lib/node_modules/n8n/node_modules/@oclif/core/lib/config/config.js:424:25)
n8n-1   |     at run (/usr/local/lib/node_modules/n8n/node_modules/@oclif/core/lib/main.js:94:16)
n8n-1   |     at /usr/local/lib/node_modules/n8n/bin/n8n:70:2
n8n-1   | EACCES: permission denied, open '/home/node/.n8n/crash.journal'
n8n-1   |     TypeError: Cannot read properties of undefined (reading 'error')
n8n-1 exited with code 0

% docker compose -f n8n/docker-compose.yml -p n8n-borzoi --env-file config/.n8n-borzoi.env run --rm -it --entrypoint /bin/sh n8n
~ $ cd .n8n/
~/.n8n $ ls -la
total 468
drwxr-xr-x    5 root     root          4096 Jun  3 01:10 .
drwxr-sr-x    1 node     node          4096 Jun  3 01:31 ..
drwxr-xr-x    2 root     root          4096 Jun  3 01:10 binaryData
-rw-r--r--    1 root     root            56 Jun  3 01:10 config
-rw-r--r--    1 root     root             0 Jun  3 01:10 crash.journal
-rw-r--r--    1 root     root        454656 Jun  3 01:10 database.sqlite
drwxr-xr-x    2 root     root          4096 Jun  3 01:10 git
-rw-r--r--    1 root     root             0 Jun  3 01:10 n8nEventLog.log
drwxr-xr-x    2 root     root          4096 Jun  3 01:10 ssh
```
