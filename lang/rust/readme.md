# Rustとは  
プログラミング言語。Firefox QuantumのCSSエンジンなどに使われている。  

基本は↓↓
$ rustc main.rs # でコンパイル. 実行可能バイナリを出力する.  
$ ./main # で実行  

RustにおけるmavenがCargo.  
$ cargo build # でbuild
$ cargo run # でrun

プロジェクト用ディレクトリ直下の.toml ファイルを参照する.  
常時はdebugビルドなので、--releaseをつけると最適なビルドをする.


# 参考  
https://rust-lang-ja.github.io/the-rust-programming-language-ja/1.6/book/
