# ロボット工学 (robotics)

ページ構成にあたって次の情報源を参考にした。

- [イラストで学ぶ ロボット工学](https://amzn.to/45Trxtw)
- [ロボット機構学](https://www.coronasha.co.jp/np/isbn/9784339045093/)
- [ロボットの種類](https://www.intel.co.jp/content/www/jp/ja/robotics/types-and-applications.html)
- [Claude🔐](https://claude.ai/chat/852725b9-763c-486c-8a69-ed8571518e7a)

## 基礎知識

## ロボット機構

### 運動学

### 静力学

### 動力学

## ハードウェア

### マニピューレータ

### アクチュエータ

- エンドエフェクタ: WIP

### センサ

- IMU: WIP
- LiDAR: WIP
- カルマンフィルタ: WIP
- 自己受容感覚（proprioception）: WIP
- interoception
- exteroception

## ソフトウェア

<!-- リアルタイムOS: WIP -->

### ROS

### DDS

ROS2に採用されているPub/Sub形式の通信プロトコル。主にUDPを用いて通信し、信頼性についてはDDSのレイヤーで担保しているが、TCPを用いる実装もある。[^youtalk_dds]
[^youtalk_dds]: [DDS (Data Distribution Service) とは](https://www.youtalk.jp/2017/05/28/dds.html)

## 設計

- バイオミメティクス: WIP

## 種類

### 移動型

#### 無人搬送車 (AGR, automated guided vehicle)

#### 自律走行ロボット (AMR, autonomous mobile robot)

##### 自己位置推定, SLAM

監視カメラやGPS等からロボットの位置情報を取得できない場合でも、ロボット自身のセンサーで位置を推定することを自己位置推定という。始めから室内等の地図情報を持っている場合はそれで十分だが、そうでない場合は、ロボット自身が移動中に地図を生成し、その中で自分の位置を推定する必要がある。これをSLAM (Simultaneous Localization and Mapping)と呼ぶ。

自己位置推定のアルゴリズムについて、ROSのパッケージ`acml`では、ベイズフィルター (bayes filter)の実装である粒子フィルター (PF, particle filter)を用いたモンテカルロ位置推定 (MCL, monte carlo localization)を採用している。このアルゴリズムでは、地図内におけるロボットの位置の確率分布を地図上の粒子として表現し、スキャン情報で位置を更新する。[^robogaku_2017]
[^robogaku_2017]: [ROSを用いた自律走行](https://robogaku.jp/news/2022/past_35_4_16.html)

`acml`には、地図形式として専有格子地図 (専有グリッドマップ, occupancy grid map)が実装されている。専有格子地図とは、環境を格子状に分割し、セル毎に障害物があるか否かの確率を求めた地図である。[^akinami_2023]
[^akinami_2023]: [【図解】占有格子地図（occupancy grid map）徹底解説](https://qiita.com/akinami/items/7bcd0818f8f76f7181d2)

##### 経路・動作計画

##### 自動運転

自動運転において、カメラやLiDAR等の車載センサーを用いた自律型自動運転に対して、車車間 (V2V)通信、路車間 (V2I)通信などを組み合わせた運転を協調型自動運転と呼ぶ。通信について次の通り整理した。[^businessnetwork_2023]
[^businessnetwork_2023]: [5.9GHz帯を「協調型自動運転」に活用へ　総務省が次世代ITSの研究会](https://businessnetwork.jp/article/12826/)

- V2X: 短距離通信を指すことが多く、技術的には主に無線LAN等を想定している。
  - V2V: 車車間通信を指す。セルラー回線を用いることもある。
  - V2I: 車とインフラ間の通信を指す。
- V2N: 広域通信を指すことが多い。

車載センサーの情報、地図情報、および通信で得た情報を統合してダイナミックマップと呼び、次の要素を含む。[^sip_2017]
[^sip_2017]: [ダイナミックマップの概念/定義](https://www8.cao.go.jp/cstp/gaiyo/sip/iinkai/jidousoukou_30/siryo30-2-1-1.pdf)

- 動的情報: 高精度3D地図情報
- 準動的情報: 交通規制予定情報, 道路工事予定情報等
- 準静的情報: 事故情報, 渋滞情報等
- 静的情報: 周辺車両, 歩行者情報, 信号情報等

<!--
- RRT: WIP
- ポテンシャルフィールド法: WIP-->

#### 歩行ロボット

- 受動歩行: WIP

### 据え置き型

## 未分類 (uncategorized)

- morphological computation: WIP
- Affordance - アフォーダンス。環境内の物体が持つ行動可能性。例えば、椅子は「座る」というアフォーダンスを持つ。
- Semantic Mapping - 意味的マッピング。環境地図に物体の意味情報を付加する技術。
- Kinesthetic Teaching - 運動学的教示。人がロボットを直接動かして、タスクを教える方法。
- Lyapunov Stability - リアプノフ安定性。非線形システムの安定性を解析する数学的手法。ロボットの制御則設計に用いられる。
- Underactuation - 劣駆動。ロボットの自由度が駆動数よりも多い状態。
- Redundancy Resolution - 冗長性解消。ロボットの冗長な自由度を利用して、タスクを達成する方法。
- Proprioceptive Feedback - 固有感覚フィードバック。ロボットの関節角度や力覚などの内部情報を利用したフィードバック制御。
- Central Pattern Generator (CPG) - 中枢パターン発生器。リズミックな運動パターンを生成する神経回路網。ロボットの周期運動の生成に利用される。
- Dynamic Stability - 動的安定性。移動ロボットが運動中に姿勢を維持できる能力。
- ゼロ・モーメント・ポイント: WIP
- ラウス・フルビッツの安定判別法: WIP
- ステッピングモーター: WIP
- ラグランジュの運動方程式: WIP
- ステップ応答と伝達関数: WIP
- H∞制御: WIP
- サーボ系における位置フィードバックとトルクフィードバック: WIP
- 近接覚センサとその原理: WIP
- インパルス応答とステップ応答とその関係: WIP
- PID制御: WIP
