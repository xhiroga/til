# docker

コンテナAPI。  

* なぜDockerか?  
同じアプリケーションの複数バージョン, 複数インスタンスを同時に走らせられるため。  

# 使い方
## Dockerイメージの作成→削除
```console:
apt-get install docker.io # インストール
docker images # イメージの確認
docker pull nginx:1.10.0 # DockerHubからイメージをインストール
docker build -t image:tag . # カレントのDockerfileからイメージ作成, 同じイメージ名+タグ名で上書き, タグ名を変えるとアップデート。
```

## Dockerコンテナーの実行→削除
```console:
docker run -d nginx:1.10.0 # -dはdetach(バックグラウンド起動), コンテナ名だけを表示する
docker run -p 55555:8080 nginx:1.10.0 # -pでポート転送。ホスト:コンテナ
docker run -it ubuntu_test # -iでインタラクティブ -tで擬似端末の割り当て（矢印とか使えるようになる）
docker ps # 起動中のコンテナの確認
docker exec -it cc21766ebf79 sh # 起動中のコンテナの中に入る
docker inspect ID # IDの頭文字とかでもOK
docker stop
```

# 参考資料
[Dockerコマンド](http://docs.docker.jp/engine/reference/commandline)

[Scalable Microservices with Kubernetes | Udacity](https://classroom.udacity.com/courses/ud615)

[Containers bring a skinny new world of virtualization to Linux](https://www.itworld.com/article/2698646/virtualization/containers-bring-a-skinny-new-world-of-virtualization-to-linux.html)

[What are containers and why do you need them?](https://www.cio.com/article/2924995/software/what-are-containers-and-why-do-you-need-them.html)
