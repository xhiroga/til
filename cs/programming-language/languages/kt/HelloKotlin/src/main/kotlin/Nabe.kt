class Nabe(vararg args: String) {
    val elements = args
}

fun main(args: Array<String>) {
    val nabe = Nabe("tofu", "connyaku", "shirataki", "renkon", "hakusai")
    for (ele in nabe.elements) {
        println(ele)
    }
}