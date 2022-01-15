// オブジェクト式（オブジェクト宣言ではなく）
val myObject = object {
    override fun toString(): String = "MyObject"
}

interface Musician {
    fun hello()
    fun play()
}

class Beatles() : Musician {
    override fun toString(): String = "Beatles"

    override fun hello() {
        println("say hello")
    }

    override fun play() {
        println("🎵")
    }
}

class Player() {
    override fun toString() = "CD Player"
    fun play(who: String) {
        if (who == "Beatles") {
            val musician = Beatles()
            // オブジェクト式で定義した匿名型のオブジェクトでもDelegateを利用できる。
            val player = object : Musician by musician {
                override fun play() {
                    // ここで $this としても "Beatles" とはならない。
                    // つまり、DelegateはあくまでInterfaceで定義された動作をbyで指定したインスタンスに任せる機能であって、
                    // Interfaceで定義されていないオブジェクトの共通の挙動については何も書き換えない（＝Anyを継承している？）

                    // this@XX については "スコープ外this式参照" とでも呼べば良さそうだ
                    println("Play $musician by ${this@Player}")
                    musician.play()
                }
            }
            player.play()
        }
    }
}

fun main(args: Array<String>) {
    // オブジェクト式編
    val hello = "hello"
    println(hello)
    println(hello.javaClass.kotlin)

    println(myObject)
    println(myObject.javaClass.kotlin) // class ThiszKt$myObject$1 (Kotlin reflection is not available) だそうです。

    // thisスコープ編
    val player = Player()
    player.play("Beatles")
}
