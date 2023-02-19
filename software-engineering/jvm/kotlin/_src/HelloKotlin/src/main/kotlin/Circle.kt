class Circle(val rad: Double) {

    // 従来のスタティックメンバー/メソッドに代わるもの。
    // companion object Companion { // 名前を宣言することもできる。
    companion object {
        val PI: Double = 3.14
    }

    // バッキングフィールドに紐つかないゲッター
    val area: Double
        get() = rad * rad * PI
}

fun main(args: Array<String>) {
    val clc = Circle(5.5)
    println(clc.area)

    println(Circle.PI)
    println(Circle.Companion.PI)

    // clc.Companion.PI　// これは間違い
}
