# IT辞典

## Initialization

 - [More \- Downloads \- Apple Developer](https://developer.apple.com/download/all/?q=Auxiliary) より、Xcodeのバージョンに即した `
Additional Tools` をインストール
 - `.dmg` ファイルをマウントした後、任意のフォルダ(ex. `~/DevTools`)にファイルをコピー
 - `Additional Tools`側で`cp`が`/bin/cp`を指すように修正(通常は不要だが、念のため)

## Build & Install

```shell
make deploy
```
