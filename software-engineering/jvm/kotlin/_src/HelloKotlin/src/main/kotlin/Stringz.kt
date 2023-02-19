fun main(args: Array<String>) {
    val str = "Kotlin is fun!"
    println(str.length)

    println("".isBlank())

    val nullz: String? = null
    // println(nullz.isBlank()) // nullable receiverでしか呼ばれません。
    val thr = 3
    println("$thr = three!")

    val message = """I
                    |  Love
                    |      You!
    """.trimMargin()

    println(message)

}