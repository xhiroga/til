# React

## Framework Rules

### Functional Components vs Class Components

Function Componentを用いる。[^functional-components]

[^functional-components]:[Reactでクラスコンポーネントより関数コンポーネントを使うべき理由5選](https://tyotto-good.com/blog/reaseons-to-use-function-component)

#### Arrow Function vs Function

アプリケーションの中で統一する。    
個人的には引数と返り値の型定義をまとめて `React.VFC` で行えるアロー関数が好み。  
function宣言には、同時に default exportできるメリットがある。

### Named Export vs Default Export

定数など、一つのファイルから複数の値をexportする場合は必然的に named export を使う。  
Componentに対しては、どちらのexportにもメリット・デメリットがある。
 - Named Exportの場合、VSCodeがリファクタリングしたコードそのままで良い。
   - Componentを新ファイルに移動するケース。
 - Default Exportの場合、1ファイルから1つのComponentしかexportできないことが担保される。

参考: [Named Export vs Default Export in ES6 \| by Alankar Anand \| Medium](https://medium.com/@etherealm/named-export-vs-default-export-in-es6-affb483a0910)


### Naming

- Component名はアプリケーションの中で一意であるべき。VSCodeがmissing importを検索する際に競合が発生するのを防ぐためである。
- Explorerから目で探しやすいように、ファイル名とComponent名は同一であるべき。
  - したがって、 `path-based-naming`[^path-based-naming] は推奨されない。
- `PascalCase` で命名する。
- 役割 + 要素で命名するのが良さそう（検討中）
  - e.g. Login(役割)Form(要素), Profile(役割)Page(要素)

[^path-based-naming]: [Structuring projects and naming components in React \| Hacker Noon](https://hackernoon.com/structuring-projects-and-naming-components-in-react-1261b6e18d76)

### Spacing

- Grid LayoutでGapを利用するのが最も良い。
- そうでなければ、チーム内にセマンティクス原理主義者(?)がいなければSpacerを利用したい。[^React で余白をどうスタイリングするか]
- 個人的には親Componentから子Componentの呼び出し時にStyleをオーバーライドする方法を採ってきたが、以下の問題がある。
  - Componentを書き換え可能なのでAtomic Designの原則的なものに反しそう。
  - margin-top or margin-bottom の議論から逃れられない。

#### Bad

```tsx
// 非推奨: 親Componentから子Componentを呼び出す際にマージンを渡す
const Container = () => {
  return (
    <div>
      this is container
      <Content style={{ marginTop: 32 }} />
    </div>
  )
}
const Content: React.FunctionComponent<React.HTMLProps<HTMLDivElement>> = (props) => {
  const { style } = props
  return <div style={style}>This is content</div>
}
```

[^React で余白をどうスタイリングするか]: [Reactで余白をどうスタイリングするか](https://zenn.dev/seya/articles/09545c7503baa4#comment-a33d0d79293f45)


### Styling

- 前提として、CSS in JSを用いる。
- CSSのクラス名がグローバルなことが起因の問題を避けるため、生のCSSは使わない。
  - BEM, OOCSS, FLOCSSなどの命名規則で解決するのも避ける。
- WebPackの設定を避けたいので、scssなどのCSS拡張の導入も避ける。

#### Object Style vs Template Style (String Style)

- 個人的にはReact Nativeで慣れているオブジェクトスタイルが好み。^[なお、[React Nativeでもテンプレートスタイル・コンポーネント生成は可能](https://styled-components.com/docs/basics#react-native)]
  - もっと言うと、（例えパフォーマンスを犠牲にしても）Component内で定義するのが読みやすくて好き。
- ただし、CSSと厳密に一致するテンプレートスタイルの方が好まれている印象。
  - CSSのシンタックスハイライトが使える。
  - Figma等から生成したCSSがコピペで使える。


```tsx
// Object Style
const styles = {
  container: {
    width: '100%'
  }
}

// Template Style / String Style
// astroturf
const className = stylesheet`
:local(.container){
    height: 100%
}
`;
```

### Adopting Style vs Generating Component

検討中。

参考: [りあクト！ TypeScriptで極める現場のReact開発 (電子版（92p）)](https://booth.pm/ja/items/1312815)

### Library

特にこだわりがないのでチームの誰かに選定してもらう。  
`styled-component`, `emotion`, `astroturf` など。
