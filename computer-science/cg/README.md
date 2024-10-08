# CG (computer graphics)

## 基礎知識

> [!NOTE] ビットマップ画像とラスタ画像って同じ？
> 同じ意味で使われるが、厳密には異なる。ラスターはラテン語で引っ掻くを意味する rodere に由来し、同じ語源を持つ言葉としては rake (熊手) や rodent (げっ歯類) がある。
> 画像を平行な線の集合と捉えてスキャンすることをラスタースキャンという。FAXの送受信はラスタースキャンを用いている。そのように表現された画像をラスタ画像と呼ぶ。
>
> ビットマップ画像はラスター画像をコンピューターで扱うための仕様といえる。元々は、フォントを画面上に表示する際のビットの割当表をビットマップと呼んだそうだ。

## 座標変換とパイプライン

### 2次元座標系

2次元図形を表すにあたって、平面上の点を(x,y)座標で表す座標系を2次元直交座標系 (two-dimensional orthogonal coordinate system)という。また、原点からの距離rと角度θの組(r,θ)で表す座標系を極座標系(polar coordinate system)という。ここで、2次元図形の変形を考える。図形の点を(x,y)からなる列ベクトルと考える。例えば水平移動や垂直移動（平行移動）と言われるについては、異なる列ベクトルの足し算と考えて良い。また拡大や縮小、回転については、行列とベクトルの積と考えて良い。

拡大・縮小や回転など、複数の変換をまとめて1つの変換にしてから列ベクトルとの積を求められたら、図形の全ての点に対して計算する量を大幅に削減できる。そのため、平行移動も行列とベクトルの積として捉えたい。そこで、(x,y)の他に定数の1をwとして加えた3次元で座標を表現し、平行移動は定数項への積として表現する。このような座標系を同次座標系(homogeneous coordinates system)という。おそらくは、座標の値をそれぞれ定数倍しても同じ座標を表す性質を"homogeneous"と形容しているのではないか。[^quora_homogeneous]また、原点から座標への直線がw=1平面と交わる点を座標とみなしていることから、射影座標(projective coordinates)とも呼ぶ。
[^quora_homogeneous]: [What does homogeneous in "homogeneous coordinate" refer to?](https://www.quora.com/What-does-homogeneous-in-homogeneous-coordinate-refer-to)

### 3次元座標系

X,Y,Zの3軸を考えたとき、Z軸を奥から手前向きに、Y軸を下から上向きに固定した際、X軸が右向き（右手系）か左向き（左手系）かの違い。または、X軸を右向きに固定した際、Y軸が上向き（右手系）か下向き（左手系）かの違い。言い換えると、座標(0,0)が左下（右手系）か左下（左手系）かの違い。

学校の数学では座標(0,0)が左下なのに対して、テレビでは座標(0,0)が左上だったことが違いを生んでいると言われている。[^stackoverflow_2011]
[^stackoverflow_2011]: [Historic reasons for Left-Handed Coordinate System](https://stackoverflow.com/questions/6698817/historic-reasons-for-left-handed-coordinate-system)

### シェーダー

2000年前後、グラフィックボードの進化に伴いCPUの処理がボトルネックとなり、オブジェクトの座標変換と照明 (Transform&Lightning)もグラフィックボードが担当するようになった。Direct7では予め用意されたシェーダーを利用するのみだったが、Direct8からはプログラミング可能なシェーダーが登場し、現在ではこれを単にシェーダーと呼ぶ。

## モデリング

### プロシージャルモデリング

炎・煙・水しぶきなど、時間によって形が変わるもの、大量にあってポリゴンでモデリングすると手間ひまがかかるものを、数式で表現すること。

<!-- Processingはプロシージャルモデリング？ -->

## レンダリング

### フォトリアリスティックレンダリング

単に3Dオブジェクトの形状をレンダリングするのではなく、画面上で写真としても正しいような（写実的な）画像を生成することをフォトリアリスティックレンダリングという。

### 隠面消去

| アルゴリズム         | 処理単位       | 隠面消去の仕組み                             | Pros                    | Cons                               |
| -------------------- | -------------- | -------------------------------------------- | ----------------------- | ---------------------------------- |
| ペインタアルゴリズム | ポリゴン       | 奥行きの順序をポリゴン単位で保持             | 単純                    | 互いに重なるポリゴンの判定が難しい |
| スキャンライン法     | スキャンライン | スキャンラインとオブジェクトの交差線分を比較 | Zバッファ法より省メモリ | 反射や屈折をサポートしない         |
| Zバッファ法          | ポリゴン       | z座標値（奥行き）をピクセル単位で比較        | GPUによる高速処理       | 反射や屈折をサポートしない         |
| 順レイトレーシング法 | レイ           | 光源から視点に辿り着いたレイを描画           | 自然                    | 無駄な計算が多い                   |
| 逆レイトレーシング法 | ピクセル       | 視点からレイをたどり交差した面を描画         | 自然                    | 計算量が多い                       |

3Dオブジェクトを描画する際、手前の面だけが見えるようにすること。凸面だけのオブジェクト単体を考えるなら、こちらを向いている面だけを描画すればよい。具体的には、法線と視点のベクトルの内積を取り、その値が正なら描画する。

奥のオブジェクトから順番に描画し、重なった箇所は手前のオブジェクトで塗りつぶすアルゴリズムを優先順位アルゴリズム、またはペインタアルゴリズムと呼ぶ。予めオブジェクトを奥行き順にソートする必要がある。オブジェクトが3すくみの場合には、オブジェクト単位で優先順位を付けることができないため、ポリゴンを2つに切断する。ペインタアルゴリズムは見えないオブジェクトまで描画するため効率が悪い。

手前のポリゴンから描画する逆ペインタアルゴリズムも存在するが、手前のポリゴンが奥のポリゴンを完全に隠している場合を除いて、やはり本来見えない箇所の描画のための計算が行われてしまう。

スクリーンを1行ごとに描画する手法として、スキャンライン法がある。[^try3dcg_render]ペインタアルゴリズムと異なり、相互に重なるポリゴンが問題にならず、半透明なオブジェクトの計算も可能。アルゴリズムはやや複雑であり、かつ時間がかかる。しかし、1行毎にディスプレイに反映できるため、後述のZバッファ法に比べてメモリの少ない環境でも採用できる。[^oshita_2019_cg05]
[^try3dcg_render]: [【はじめての3DCG】：スキャンライン](https://www.asahi-net.or.jp/~qb3k-kwsk/3dcg/know/render/render11.html)
[^oshita_2019_cg05]: [コンピューターグラフィックスＳ](http://www.ha.ai.kyutech.ac.jp/lecture/cg/cg05_rendering_s.pdf)

ペインタアルゴリズムはポリゴンの分割やオブジェクトのソートが必須であり、アルゴリズムがやや複雑となり、おそらく並列計算との相性も悪い。そこで、全てのオブジェクトのピクセルについて、そのピクセルが暫定的に最も手前であった場合にバッファの色を更新するようなアルゴリズムがあり、これをZバッファ法と呼ぶ。Zバッファとは、暫定的に最も手前にあるオブジェクトのz座標値をピクセルごとに持つバッファである。GPUのサポートがあることから、広く利用されている。

レイトレーシングについては[シェーディング](#シェーディング)の節で述べる。

### 隠線消去

3Dオブジェクトを描画する際、視点から見えない部分の線を表示しない処理を隠線消去という。20世紀フォックスのロゴを思い浮かべると良い。

### シェーディング

シェード (陰, shade)とシャドウ (影, shadow)は元々は同じ言葉だったとされている。[^etymonline_shade]中英語以降、シェードが光の届きづらい暗い部分を指すのに対して、シャドウは光を遮られたことで浮き出た暗い形を指すようになった。
[^etymonline_shade]: [shade(n.)](https://www.etymonline.com/word/shade)

シェーディングは照明の計算によってなされる。照明の計算を行うモデルをシェーディングモデルや照明モデルという。シェーディングのみを独立して行う手法としては、ポリゴン単位で光の強さを計算して色を決めるフラットシェーディングや、頂点毎に光の強さを計算し、その結果を線形補間してポリゴンに適用するグーローシェーディングがある。

反射光を考慮するにあたって、現実の物体の表面は平らではないため、反射と乱反射のいずれもが起きる。これを表現するために、光の入射位置、入射方向、反射角に関する関数を定義する方法をBRDF (双方向反射率分布関数, Bidirectional Reflectance Distribution Function)という。[^oshita_2019_cg12]
[^oshita_2019_cg12]: [コンピュータグラフィックスＳ 第12回](http://www.cg.ces.kyutech.ac.jp/lecture/cg/cg12_mapping.pdf)

隠面消去で用いられるレイトレーシングは、同時にシェーディングを扱うことができる。光源から視点までの光の経路を、視点から逆に辿ることを逆レイトレーシング、あるいは単にレイトレーシングという。隠面消去に関しては、ピクセルごとに逆レイトレーシングを行い、レイが交差した面の色をピクセルに描画すれば良い。レイが交差した物体が鏡やガラスのように反射・屈折を起こす場合、反射方向と屈折方向にレイを枝分かれさせる。隠面消去に加えて反射や屈折をシミュレーションするのであれば、レイの追跡は反射や屈折を起こさない面と交差したところで打ち切って良い。こうした物体を拡散反射面という。

### 影付け（シャドウイング）

影付けにも隠面消去のアルゴリズムが応用されている。レイトレーシング法、スキャンライン法、Zバッファ法を用いた方法がある。

### マッピング

3Dオブジェクトに模様や画像を貼り付けて表示することをテクスチャマッピングと呼ぶ。プロジェクションマッピングの要領でテクスチャを表示することを投影テクスチャマッピング、または単に投影マッピングと呼ぶ。プロジェクターの前を歩いても画像が人に着いてこないように、テクスチャはオブジェクトの動きに追随しない。オブジェクトに模様を追随させるには、座標を用いたマッピングが行われる。

模様の貼付けではなく凹凸をマッピングする手法をバンプマッピングと呼ぶ。エッチング加工のような手法と言えそうだ。

反射による周囲の映り込みをマッピングにより擬似的に再現する方法を環境マッピングと呼ぶ。スーパーマリオ64でメタルマリオに花の写真をマッピングしていた手法ではないか。[^n64_2022]
[^n64_2022]: [Here is the texture for the Metal Mario reflection on Mario64](https://www.reddit.com/r/n64/comments/svgtwm/here_is_the_texture_for_the_metal_mario/)

テクスチャを3次元空間の模様として表現することをソリッドマッピングと呼ぶ。角や断面においてもマッピングが自然になる効果がある。

### 大域照明計算 (global illumination calculation)

太陽などの光源からの直射光のみを考慮する考え方を局所照明と呼ぶのに対して、間接光まで考慮することを大域照明 (global illumination)と呼ぶ。大域照明のレンダリング方程式は、次の通り大別される。

- ラジオシティ法
- モンテカルロ法
- マルコフ連鎖モンテカルロ法

ラジオシティ法では、物体を小さなパッチに分割し、それぞれのパッチの単位面積あたりの放射を未知数とした方程式を解くことで明るさを求める。[^nishita_2014]シーン内の物体の形状が変わらなければ、カメラの位置が変わっても明るさを再計算しないで良い点がメリットである。
[^nishita_2014]: [Radiosity Method](http://nishitalab.org/user/nis/ourworks/radiosity/radiosity.html)

隠面消去やシェーディングのためのレイトレーシングでは、レイの追跡を拡散反射面で打ち切った。レイをさらに追跡することで、直接光と間接光を統一的に扱うことが可能である。しかしながら、特に拡散反射面における光の反射をすべて追跡することはできない。そこで、追跡する光をランダムに選ぶことにする。このようにモンテカルロ法を用いるレイトレーシングの手法を、特にパストレーシング (経路追跡法, path tracing) という。

パストレーシングでは、カメラに写っている面と光源の間の光の経路を計算し、光源まで運よくたどり着けば寄与を計算する。この際、モンテカルロ積分を用いて各経路の寄与を平均化し、最終的なピクセルの輝度を求める。この手法は多くの経路を追跡し、その結果を平均化することでリアルな画像を生成することができる。

同じくモンテカルロ法を用いる手法として、フォトンマッピングがある。光源からと視線からの2つのレイトレーシングを行ったうえで、それらの経路が近くにあるときは結んで1つの経路とみなす。

## グラフィックス

### 可視化

MRIで撮影したデータなどの3次元のボリュームデータを可視化する技法として、ボリュームレンダリングがあり、その代表的な手法にレイキャスティングがある。物体の表面かどうかにかかわらず視線からのレイが通った経路を取得し、その経路上のボリュームには不透明度を設置することで、3次元のオブジェクトが透けている状態で観測ができる。

### 自由視点画像生成

複数の視点の画像を基に任意の視点からの画像を生成すること。空間中の全ての地点における全ての光線を求める手法、対称のオブジェクトの3次元形状を復元する手法等がある。

画像を点群データに変換し、さらに点群を重なり合わない三角形の集合に分割してメッシュを作り、写真の色情報を元にテクスチャを貼り付ける形で、多角形のポリゴンを生成する手法をフォトグラメトリという。[^yukis0111_2024]
[^yukis0111_2024]: [3D生成手法の比較(フォトグラメトリー & NeRF & 3D Gaussian Splattig)](https://qiita.com/yukis0111/items/87359b30ddef2856d3fa)

ニューラルネットワークを用いた手法として、3次元の空間における位置と視野角を入力情報として、各点に物体がある度合いと色の分布を求めるNeRFが知られている。[^ben_2020]NeRFではオブジェクトごとにニューラルネットワークのモデルが必要となる。
[^ben_2020]: [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934)

## アニメーション

## References

[コンピュータグラフィックス](https://amzn.to/3xirgU8)
