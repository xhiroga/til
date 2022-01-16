data class Member(val id: Int, val nickname: String) {}

// data object Community(val members: Array<Member>) // シングルトンなデータ型は作れない
class Community(val id: Int) {} // 別にデータクラスじゃなくてもプライマリコンストラクタ内でのプロパティ宣言はできる

fun main(args: Array<String>) {
    val hiro = Member(1, "hiro")
    println(hiro.toString()) // Member(id=1, name=hiro)

    val hiroo = Member(2, "hiro")
    val hiro2 = Member(1, "hiro")

    println(hiro == hiroo)
    println(hiro == hiro2)
    println(hiro === hiro)

    val hiroshi: Member = hiro.copy(id = 3)
    println(hiroshi)

    // 分解宣言（JavaScriptの... スプレッド演算子みたいなもの）
    val (hiroshiId, hiroshiName) = hiroshi
    println("id: $hiroshiId, name: $hiroshiName")

    // data object式はないらしい。
    // val greet = data object{
    //     val language = "English"
    // }
}