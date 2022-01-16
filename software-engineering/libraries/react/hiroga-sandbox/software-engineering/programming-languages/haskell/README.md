# Haskell

## 使い方
* 対話実行  
$ ghci # インタプリタを起動, Ctrl+D で終了  
* コンパイラの実行  
$ ghc -o hello hello.hs # 実行可能ファイルが生成される  

## 仕様
(チュートリアルの観察メモ)  
* 関数の返り値がtupleであることが多いため、fst関数を多用する。
* letでローカルスコープの変数を作成し、in bodyで使用する。   
* String型は存在しない。''で表されるCharと、""で表されるCharのリストがある。  

## 参考
* Haskell
https://www.haskell.org/
