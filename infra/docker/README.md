# docker

コンテナAPI。  

* なぜDockerか?  
同じアプリケーションの複数バージョン, 複数インスタンスを同時に走らせられるため。  

# 使い方
```Console:
apt-get install docker.io # インストール
docker images # イメージの確認
docker pull nginx:1.10.0 # DockerHubからイメージをインストール
docker run -d nginx:1.10.0 # -dはdetach(バックグラウンド起動)
docker inspect ID # IDの頭文字とかでもOK
docker stop
```

# 参考資料
https://classroom.udacity.com/courses/ud615  
https://www.itworld.com/article/2698646/virtualization/containers-bring-a-skinny-new-world-of-virtualization-to-linux.html  
https://www.cio.com/article/2924995/software/what-are-containers-and-why-do-you-need-them.html  
