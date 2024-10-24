# 暗号理論 (cryptography)

ページ構成の際、次の情報源を参考にした。

- [暗号理論入門 原書第3版](https://amzn.to/4fA1TP0)（[丸善出版](https://www.maruzen-publishing.co.jp/item/b294238.html)）
- [Claude🔐](https://claude.ai/chat/eb58612f-ac93-4fac-b4dd-13b50a3b93f4)

## DESアルゴリズム

## AWS暗号化アルゴリズム

## 公開鍵方式

## デジタル署名

### ヒステリシス署名

ブロックチェーンのように、デジタル署名の連鎖によってデータの改竄検知、連続性の担保を行う技術。PoWのような合意形成の仕組みは無いため、ブロックチェーンとは違い中央サーバーが必要な点が大きく異なる。日本の大学および日立製作所による共同研究。[^hitachi_2003]
[^hitachi_2003]: [ヒステリシス署名](https://web.archive.org/web/20090906015858/http://www.hitachi.co.jp/Div/jkk/glossary/0410.html)

## 認証

### ゼロ知識証明

## 秘密の分散共有

### シャミアの秘密分散法

暗号化された情報の鍵の扱いを考える。鍵が複数箇所に保存されていると漏洩しやすく、1箇所だと紛失しやすい。n個の鍵のうちk個があれば鍵（秘密情報）が復元できると良い。これを(k,n)しきい値法という。

(k,n)しきい値法の具体的なアルゴリズムにシャミアの秘密分散法がある。k-1次多項式はk個の点の座標によって復元できる。このとき、多項式の切片は次数に依らないため、秘密情報を切片として扱うと便利である。[(k,n)しきい値法とシャミアの秘密分散法](https://manabitimes.jp/math/1181)も参照。

### 秘密計算

センシティブな数値の計算が大量にあり、外部に委託したいとする。データを共有する際に暗号化しても、計算の際に復号化するのでは、外部の事業者の信頼性を担保することにコストがかかる。そこで、暗号化したままで計算できると便利である。そのような計算を秘密計算という。利用される暗号として、準同型（じゅんどうけい）暗号がある。準同型は射像の分類で、全く同型とは限らないが、特定の演算についてはかけ算でいう分配法則が成り立つようなものをいう。
