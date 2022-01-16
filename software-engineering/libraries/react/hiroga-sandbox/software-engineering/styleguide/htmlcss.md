# HTML CSS Style Guides

## HTML

### Meta Rules

#### File Extension

`.htm`と`.html`に本質的な違いは無い。  
Chrome で Web ページを保存する際のデフォルト拡張子が`.html`であることに合わせ、私は`.html`を用いる。

[Should you use .htm or .html file extension? What is the difference, and which file is correct? [closed]](https://stackoverflow.com/questions/1163738/should-you-use-htm-or-html-file-extension-what-is-the-difference-and-which-f)

## CSS

### Style Rules

#### Unit

前提として、ブラウザにおけるズームには2種類ある。

1. `Ctrl` + `+/-` で提供される、フォントや要素をまとめた拡大・縮小
2. Chromeの場合[Appearance - Font Size](chrome://settings/appearance)から提供される、スタイルシートのルートのフォントサイズの変更。

フォントサイズに応じて拡大したい、おそらくわずかしかいないユーザーを尊重したい場合のみ `rem` を用いればよいが、 `px` で十分。

フォーマットにおけるルールとして、0の場合は単位を付けない。

##### References

- [css \- Why em instead of px? \- Stack Overflow](https://stackoverflow.com/questions/609517/why-em-instead-of-px)
- [html \- Should I use px or rem value units in my CSS? \- Stack Overflow](https://stackoverflow.com/questions/11799236/should-i-use-px-or-rem-value-units-in-my-css)
