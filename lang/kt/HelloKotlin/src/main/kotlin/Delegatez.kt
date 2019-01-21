abstract class Greeter(val name: String) {
    open fun hello() = println("hello, $name")
}

class SpanishGreeter(name: String) : Greeter(name) {
    override fun hello() = println("hola! $name")
}

// DelegateできるのはInterfaceだけ。したがって、レシーバーを必要とする処理を委譲することはできない。
// class countableGreeter(val greeter): Greeter by greeter{}


// Interfaceはメンバーの値を持たないので名前の後でコンストラクタを利用することはない
interface GreetBot {
    fun hello()
    fun bye()
}

class HiGreetBot() : GreetBot {
    override fun hello() {
        println("hello!")
    }

    // interfaceを継承したクラスでは、delegateを用いない限りは持っているメソッドを全て継承する必要がある
    override fun bye() {
        println("bye")
    }
}

class CountableGreet(val greetBot: GreetBot) : GreetBot by greetBot {
    var count = 0

    override fun hello() {
        count++
        greetBot.hello()
    }
}

fun main(args: Array<String>) {
    // 継承ではなく処理の委譲（コンポジション）をする場合に有効。
    // 例えば契約をちょっと扱いやすくしたオブジェクトが必要な場合とか。
    val bot = HiGreetBot()
    val countableBot = CountableGreet(bot)
    countableBot.hello()
    println(countableBot.count)
}
