# systemdでtightvncserverを動かす

# Usage
```bash
vi /etc/systemd/system/vncserver@.service

#  MEMO: .serviceの設定が厄介で、とりあえず以下に従えば動くことを検証した。
[Unit]
Description=Remote desktop service (VNC)
After=syslog.target network.target
 
[Service]
Type=forking
User=pi
PAMName=login
PIDFile=/home/pi/.vnc/%H:%i.pid
ExecStartPre=-/usr/bin/vncserver -kill :%i > /dev/null 2>&1
ExecStart=/usr/bin/vncserver -depth 24 -geometry 1280x800 :%i
ExecStop=/usr/bin/vncserver -kill :%i
 
[Install]
WantedBy=multi-user.target

# 機能しているか確認
systemctl start vncserver@1.service

# startupに設定
systemctl daemon-reload
systemctl enable vncserver@1.service
```

# Reference
https://www.raspberrypi.org/forums/viewtopic.php?f=66&t=123457&p=830506
https://www.orsx.net/archives/5344