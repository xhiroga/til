data class Lunch(val name: String)

fun main(vararg args: String) {
    val rng = 1..99
    println(rng.javaClass.kotlin)

    val strrng = "a".."z"
    println(rng.javaClass.kotlin)

    // rangeOf() という関数はない。
}