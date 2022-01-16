fun main(args: Array<String>) {
    // 拡張関数にLambda式や匿名関数を代入しようと思ったが、そもそもレシーバーをthisで受け取れない
    // したがって拡張関数を定義するには関数宣言と同じようにするしかない？
    fun Int.isOdd(): Boolean = (this % 2 != 0)
    println(5.isOdd())
}

// 拡張プロパティはmainの外側でしか定義できない（スコープがややこしくなるから？）
val Int.isEven: Boolean
    get() = (this % 2 != 0)

// 拡張関数はmainの外側でも内側でも定義できる。
fun Int.is3Times(): Boolean = (this % 2 != 0)

fun sub() {
    // println(5.isOdd()) // エラーになる。拡張関数はローカルスコープに限定される
}