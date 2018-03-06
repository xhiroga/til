# Kubernetes(クーベネティス)

ScalableなMicroservicesのためのインフラストラクチャー  
アプリケーションをモジュールに分割し(portability)、デプロイを簡単にし(deployability)、一部分だけをスケールできるようにする(scalability)。  

SaaSアプリケーション開発には12の原則からなる方法論がある。  
https://12factor.net/ja/  

JWT(ジョート)... JSON Web Tokens  
2者間で安全に情報をやり取りするための規格。base64EncodeされたHeader+Payloadと、それらに秘密鍵を添付して作成したハッシュ(=サイン)からなる。  
jwt.ioで簡単にEncode/Decode可能。  
https://jwt.io/  



## GCP
以下の二つを有効にする必要がある。  
* Google Compute Engine API  
* Google Kubernetes Engine API  






# 参考資料
https://classroom.udacity.com/courses/ud615  
* Udacityで紹介しているResource一覧
https://classroom.udacity.com/courses/ud615/lessons/7826112332/concepts/80841806450923  
* k8sのサンプル  
https://github.com/udacity/ud615  
