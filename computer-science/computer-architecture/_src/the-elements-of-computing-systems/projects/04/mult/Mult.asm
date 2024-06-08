// Original: https://people.duke.edu/~nts9/asmtoml/Mult.asm
// This file is part of www.nand2tetris.org
// and the book "The Elements of Computing Systems"
// by Nisan and Schocken, MIT Press.
// File name: projects/04/Mult.asm

// R0*R1の結果をR2に格納する。
// ここで、R0, R1, R2...はRAM[0],RAM[1],RAM[2]を指す...と本には書かれているが、実際にWebIDEで`R0`を呼び出すと`Error loading memory: Line XX: expected end of input` が発生する。
// Mは変数というよりは、A命令(@xxx)で指定したメモリアドレスの値を取得する関数

@2
M=0 // R2を初期化

@0
D=M
@END    // Hackの機械語では、変数を事前に宣言する必要がない。LOOP終了後の(END)を指す。
D;JEQ   // Jump EQualの頭文字。R0が0なら積であるR2も0なので、わざわざ計算しない、ということ。

@1
D=M
@END
D;JEQ

// （判定用に）元の値を保持するため、ループごとに-1するための値を異なるメモリアドレスに移す。
// Hack機械語は同時に複数のメモリをレジスタで参照できないため、Dレジスタを経由して値を渡す必要がある。
@0
D=M
@3
M=D

(LOOP)
@1	//GET 2ND NUM
D=M	//D HAS 2ND NUM

@2	//GO TO FINAL ANSWER BOX
M=D+M	//RAM[2] NOW HAS 2ND NUMBER + ITS PREVIOUS VALUE

@3	//GET 1ST NUM
M=M-1	//1ST NUM-1

D=M
@LOOP
D;JGT   // M;JGT と書きたくなるところだが、Aレジスタには`@LOOP`が代入されているため不可。

(END)
@END
0;JMP   // 無限ループにしないと無限に次の行に遷移してしまう仕様
