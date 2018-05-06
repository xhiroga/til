# dd
ファイルシステムを経由せずにデータのコピーが可能なコマンド。dataset difinitionの略。  
IBMのメインフレームのDD文に由来するため引数の構文の雰囲気が異なっている。  

cpとは異なり、巨大ファイルの一部だけのコピーや、コピーの中断再開などが可能。  
代わりにbuffer sizeをマニュアルで指定する必要があり、結果cpより速度が落ちることもある。
cpと異なり標準出力に流せるので、sshを経由しての受け渡しも可能である。  
[What is the difference in using cp and dd when cloning USB sticks?](https://askubuntu.com/questions/751193/what-is-the-difference-in-using-cp-and-dd-when-cloning-usb-sticks)

## Usage
```
# ランダムなファイルを作成する。  
dd if=/dev/random of=random.dat count=1 bs=1024
```