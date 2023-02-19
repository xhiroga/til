# React Native

## Framework Rules

### Styling

- 個人的にはInline Styleを見やすく感じる。またコンポーネント生成はしない。
- パフォーマンスを考慮するなら、Outside Styleにする。
  - [オブジェクトかStylesheetかでパフォーマンスに大きな違いはない](https://stackoverflow.com/questions/38958888/react-native-what-is-the-benefit-of-using-stylesheet-vs-a-plain-object)ので、import不要で書けるオブジェクトを採用する。

#### Inline Style

```tsx
<View style={{ flex: 1, backgroundColor: "#fff" }} />
```

#### Outside Style

```tsx
const containerStyle = { flex: 1, backgroundColor: '#fff'}

<View style={containerStyle} />
```

## Misc

### React Native vs Flutter

- コミュニティの大きさ、CSS が使えること、Web のサポートではReact Nativeが優る。
- Navigation や State の管理が組み込まれている点ではFlutterが優る。

参考: [React Native Web vs\. Flutter web \- LogRocket Blog](https://blog.logrocket.com/react-native-web-vs-flutter-web/)
s