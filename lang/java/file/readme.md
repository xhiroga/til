# ファイル
操作にはパスを与えてインスタンス化下~Reader/~Writerオブジェクトを用いる。  


# ファイルの読み込み
BufferedReaderインスタンスのread()で1行づつ読み取る。  
コンストラクタの引数にFileReaderを渡すこと。  

FileReaderインスタンスのraed()で１文字づつ読み取ってもよい。  
引数にcharを渡せばそこに、なければint型の返り値として読み取った文字を書き込む。  
いずれの場合でも返り値-1でEOLを判定する。  


# バイナリファイル
FileInputStreamなどを使用する。  
read()の引数に渡すのがbyte型である点のみ異なる。


# 一時ファイルを作る
File.createTempFile() で作成可能。  
OSデフォルトの一時フォルダ内に作成すること,ランダムなファイル名にすることを保証する。  
プロセス終了後すぐにファイルがなくなる訳ではない。  

# 参考
https://docs.oracle.com/javase/jp/7/api/java/io/File.html#createTempFile(java.lang.String,%20java.lang.String)
