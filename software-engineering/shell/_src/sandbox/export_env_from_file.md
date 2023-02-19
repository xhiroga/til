# ファイルから環境変数を設定する

```bash
# set: シェルの設定を変更する。 -a:allexport(全てのシェル変数を同時に環境変数にセットする)
set -a
# .: sourceコマンドと同じ
. .env
# allexportを無効化する
set +a
```