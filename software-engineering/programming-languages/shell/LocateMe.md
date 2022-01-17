# LocateMe
MacOSの位置情報をコマンドラインで表示するアプリケーション


# Setup
```console
curl https://jaist.dl.sourceforge.net/project/iharder/locateme/LocateMe-v0.2.1.zip -o /tmp/LocateMe.zip\
&& unzip -q /tmp/LocateMe.zip -q -d /tmp/LocateMe \
&& sudo mv /tmp/LocateMe/LocateMe /usr/bin/LocateMe \
$$ rm -rf /tmp/LocateMe.zip /tmp/LocateMe
```


# Usage
```console
LocateMe
LocateMe -g | xargs -0 open # Open My Location in GoogleMap
```


# Reference
https://sourceforge.net/projects/iharder/?source=typ_redirect