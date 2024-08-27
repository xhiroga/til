# C++

## 使い方
g++でコンパイルし、実行可能ファイル(.out)を作成する。  

```shell
$ g++ space.cpp -o space.out  
```

-o を指定しない場合は"a.out"に固定で出力される(すでにあれば上書き)


## C++プログラムの構成

preprocessor directive and main function
* # ... preprocessor directive
* main()... 0を返す決まり(返さなくても動く)  
  - int main() {}


## キーワードの読み方

* #include ... 指定したライブラリのdirectiveを与える  
* <> ... 標準ライブラリを探せ  
* "" ... currentをまず探し、なければ標準ライブラリを探せ  
* std::cout ... coutをstdライブラリから使うよ  
* using namespace std; ... std::を省略できる  


## 標準入出力

isotreamライブラリのstdモジュールを使う. bashのリダイレクトと書き方一緒.  
* std::cout << "Hey" << "Jude.\n";  
  - 引数連続で取れる.  
  - "\n"がないと出力しない.  
* std::cin >> name;  
  - スペースが勝手にセパレータ扱いされる. そうして引数を二つ渡した場合、余分な方は次のcinでそのまま使われる.  
  - スペース込みで受け取るには getline(cin,var);  
  - 数値にしたければ stringstream(var) >> num  ;

## File IO

fstreamライブラリ  
create stream -> streamがfileをopenする流れ  
文字列もライブラリに入っている。 <string>  


## Header Files

includeするライブラリを別ファイルに列挙したもの.自作する.  
Header File自体のincludeは""で囲う(currentから持ってくるから)
how do a task & what to do


## 面白かった文法
* 引数をまとめて宣言できる！
  - std::string name, address, phone;

sstream のstringstreamを使えばStringがnumericになる
getlineで文字列として受け取ったものは、stringstreamで数値にする。
