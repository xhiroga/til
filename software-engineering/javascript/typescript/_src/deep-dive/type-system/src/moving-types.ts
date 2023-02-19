// 型アノテーションのtypeof
const japaneseGreeting = "おはようございます"
let todaysGreeting: typeof japaneseGreeting // これ string じゃなくて文字列リテラルを返すのか...やはりコンパイル時のtypeofと実行時のtypeofは効果が異なる。
todaysGreeting = "おはようございます"
// todaysGreeting = "Good Morning!"

// 型関連の記述のtypeofで取得する文字列がストリングリテラルなのは、辞書のキーでも同じこと。
const colors = {
    red: 'red',
    blue: 'blue'
}
type Colors = keyof typeof colors   // red | blue
