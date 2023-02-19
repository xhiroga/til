# 言語処理100本ノック

## Development

```shell
poetry install

brew install mecab
brew install mecab-ipadic
brew install xz

make
```

## Note

第5章を`cabocha-python`で実装しようと試みたが（[ch05.ipynb](./ch05.ipynb)）、SwingObjectの扱いに不慣れなせいか上手く行かなかった。

## References and Inspirations

- [言語処理100本ノック 2020 \(Rev 2\) \- NLP100 2020](https://nlp100.github.io/ja/)
- [VS CodeでJupyterしてみよう：Visual Studio Codeで快適Pythonライフ（1/2 ページ） \- ＠IT](https://atmarkit.itmedia.co.jp/ait/articles/2108/06/news030.html)
- [MeCab: Yet Another Part\-of\-Speech and Morphological Analyzer](https://taku910.github.io/mecab/)
- [MacにMeCabを利用できる環境を整える \- Qiita](https://qiita.com/paulxll/items/72a2bea9b1d1486ca751)
- [Python\-機械学習\-自然言語処理\-言語処理100本ノック 2020 カテゴリーの記事一覧 \- ギークなエンジニアを目指す男](https://www.takapy.work/archive/category/Python-%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92-%E8%87%AA%E7%84%B6%E8%A8%80%E8%AA%9E%E5%87%A6%E7%90%86-%E8%A8%80%E8%AA%9E%E5%87%A6%E7%90%86100%E6%9C%AC%E3%83%8E%E3%83%83%E3%82%AF%202020)

### [ch05.ipynb](./ch05.ipynb)

- [CaoboCha: Yet Another Japanese Dependency Structure Analyzer](https://taku910.github.io/cabocha/)
-  [cabocha/python at master · ikegami\-yukino/cabocha](https://github.com/ikegami-yukino/cabocha/tree/master/python)
- [【Mac】Python の CaboCha をインストールして係り受け解析を行う – notemite\.com](https://notemite.com/python/python-cabocha/)
- [CaboCha & Python3で文節ごとの係り受けデータ取得 \- Qiita](https://qiita.com/ayuchiy/items/c3f314889154c4efa71e)
