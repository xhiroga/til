# User

# Memo
* `USER hiroaki`コマンドを実行しても、そのユーザーのホームディレクトリに自動的に移動する訳ではない。  
```
Step 4/7 : USER hiroaki
 ---> Using cache
 ---> 504c20fa616f
Step 5/7 : RUN pwd
 ---> Running in 0eae6428b42f
/
```