# 動的計画法（依存関係の可視化）

## 編集距離

`MONEY`を`LOVE`に変身させる場合で、編集距離の依存関係をフローチャートで示す。なお、最短距離を求めるタスクを考え、複数の矢印がある場合は最小の値を取る。

```mermaid
flowchart TD

m5l4(MONEY→LOVE: 3)
m4l4(MONE→LOVE: 2)
m5l3(MONEY→LOV: 4)
m3l4(MON→LOVE: 3)
m4l3(MONE→LOV: 3)
m5l2(MONEY→LO: 4)
m2l4(MO→LOVE: 3)
m3l3(MON→LOV: 2)
m4l2(MONE→LO: 3)
m5l1(MONEY→L: 5)
m1l4(M→LOVE: 4)
m2l3(MO→LOV: 2)
m3l2(MON→LO: 2)
m4l1(MONE→L: 4)
m1l3(M→LOV: 3)
m2l2(MO→LO: 1)
m3l1(MON→L: 3)
m1l2(M→LO: 2)
m2l1(MO→L: 2)
m1l1(M→L: 1)

%% m+l=9
m4l4--+1-->m5l4
m4l3--+(Y!=E)-->m5l4
m5l3--+1-->m5l4

%% m+l=8
m3l4--+1-->m4l4
m3l3--+(E!=E)-->m4l4
m4l3--+1-->m4l4

m4l3--+1-->m5l3
m4l2--+(Y!=V)-->m5l3
m5l2--+1-->m5l3

%% m+l=7
m2l4--+1-->m3l4
m2l3--+(N!=E)-->m3l4
m3l3--+1-->m3l4

m3l3--+1-->m4l3
m3l2--+(Y!=V)-->m4l3
m4l2--+1-->m4l3

m4l2--+1-->m5l2
m4l1--+(Y!=O)-->m5l2
m5l1--+1-->m5l2

%% m+l=6
m1l4--+1-->m2l4
m1l3--+(O!=E)-->m2l4
m2l3--+1-->m2l4

m2l3--+1-->m3l3
m2l2--+(N!=V)-->m3l3
m3l2--+1-->m3l3

m3l2--+1-->m4l2
m3l1--+(E!=O)-->m4l2
m4l1--+1-->m4l2

m4l1--+1-->m5l1

%% m+l=5
m1l3--+1-->m1l4

m1l3--+1-->m2l3
m1l2--+(O!=E)-->m2l3
m2l2--+1-->m2l3

m2l2--+1-->m3l2
m2l1--+(N!=O)-->m3l2
m3l1--+1-->m3l2

m3l1--+1-->m4l1

%% m+l=4
m1l2--+1-->m1l3
m1l1--+(O!=O)-->m2l2
m2l1--+1-->m3l1

%% m+l=3
m1l1--+1-->m1l2
m1l1--+1-->m2l1
```

## 最長部分増加列

`3141592`の最長部分増加列（longest increasing subsequence）を求める際の依存関係をフローチャートで示す。描いてから気づいたけど、超見づらいので（依存関係が一方通行なんだな）という雰囲気だけ感じてください。かっこ内の数字は増加数、εは番兵である。矢印が2本ある場合、増加数が多くなる方を取る。

```mermaid
flowchart TD

LISe3141592("LIS(ε3141592): ε3459(4)")
LIS3141592("LIS(3141592): 3459(3)")
LISe141592("LIS(ε141592): ε1459(4)")
LIS341592("LIS(341592): 3459(3)")
LIS141592("LIS(141592): 1459(3)")
LISe41592("LIS(ε41592): ε159(3)")
LIS41592("LIS(41592): 459(2)")
LIS31592("LIS(31592): 359(2)")
LIS11592("LIS(11592): 159(2)")
LISe1592("LIS(ε1592): ε159(3)")
LIS4592("LIS(4592): 459(2)")
LIS3592("LIS(3592): 359(2)")
LIS1592("LIS(1592): 159(2)")
LISe592("LIS(ε592): ε59(2)")
LIS592("LIS(592): 59(1)")
LIS492("LIS(492): 49(1)")
LIS392("LIS(392): 39(1)")
LIS192("LIS(192): 19(1)")
LISe92("LIS(ε92): ε9(1)")
LIS92("LIS(92): 9(0)")
LIS52("LIS(52): 5(0)")
LIS42("LIS(42): 4(0)")
LIS32("LIS(32): 3(0)")
LIS12("LIS(12): 12(1)")
LISe2("LIS(ε2): ε2(1)")
LIS2("LIS(2): 2(0)")
LIS1("LIS(1): 1(0)")
LISe("LIS(ε): ε(0)")


%% 7→8桁
LIS3141592--+1-->LISe3141592
LISe141592--+0-->LISe3141592

%% 6→７桁
LIS341592--+0-->LIS3141592

LIS141592--+1-->LISe141592
LISe41592--+0-->LISe141592

%% 5→6桁
LIS41592--+1-->LIS341592
LIS31592--+0-->LIS341592

LIS41592--+1-->LIS141592
LIS11592--+0-->LIS141592

LIS41592--+1-->LISe41592
LISe1592--+0-->LISe41592

%% 4→5桁
LIS4592--+0-->LIS41592
LIS3592--+0-->LIS31592

LIS1592--+0-->LIS11592

LIS1592--+1-->LISe1592
LISe592--+0-->LISe1592

%% 3→4桁
LIS592--+1-->LIS4592
LIS492--+0-->LIS4592

LIS592--+1-->LIS3592
LIS392--+0-->LIS3592

LIS592--+1-->LIS1592
LIS192--+0-->LIS1592

LIS592--+1-->LISe592
LISe92--+0-->LISe592

%% 2→3桁
LIS92--+1-->LIS592
LIS52--+0-->LIS592

LIS92--+1-->LIS492
LIS42--+0-->LIS492

LIS92--+1-->LIS392
LIS32--+0-->LIS392

LIS92--+1-->LIS192
LIS12--+0-->LIS192

LIS92--+1-->LISe92
LISe2--+0-->LISe92

%% 1→2桁
LIS2--+0-->LIS92
LIS2--+0-->LIS52
LIS2--+0-->LIS42
LIS2--+0-->LIS32
LIS1--+1-->LIS12
LIS2--+0-->LIS12
LIS2--+1-->LISe2
LISe--+0-->LISe2
```