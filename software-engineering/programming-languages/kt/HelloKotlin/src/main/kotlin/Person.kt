open class Person {
    var name: String = ""
}

fun main(args: Array<String>) {
    // クラスはインスタンスの元となるだけではなく、型を作り出します...普通じゃない？
    val taro = Person()
    taro.name = "Taro"
    println(taro.name)

    val hiro = Hero("hiro")
//    hiro.name = "takahiro" // これはできない。えらい！
    hiro.show()

    println(hiro is Hero)
    Teacher.say()
}

// こんなコンストラクタ定義もできます。
open class Hero(val name: String) {
    // クラス内のメソッド宣言であれば、クラス内のローカルスコープの関数オブジェクトをそのまま関数宣言の本体として突っ込むことが可能
    val func = fun() { println(name) } // どうもオブジェクト内の値を参照するのにthisは不要らしい。

    open fun show() = func

    val initial: Char
        // 急にインデント出てきた！
        get() = name[0]
}

// クラスの継承をするときは、継承先のプライマリコンストラクタの引数にvalをつけることはできない
class Villan(val darkName: String, name: String) : Hero(name) {
    // overrideする場合は、どちらもメソッド宣言でメソッドを作成しないとできないようだ。
//    override fun show() {
//        println("$darkName, $name")
//    }
}

// オブジェクトもクラスを継承することができる
// 作成した瞬間からオブジェクト
object Teacher : Person() {
    fun say() {
        println("say hello")
    }
}

