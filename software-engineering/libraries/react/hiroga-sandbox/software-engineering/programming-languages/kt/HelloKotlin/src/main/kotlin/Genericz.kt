class Containers<E>(vararg val elements: E)

fun <E> generateContainer(vararg elements: E) = Containers(*elements)
fun <E> generateContainer2(vararg elements: E): Containers<E> = Containers(*elements) // 当然、返り値の型を宣言できる

// Type Annotationは関数名よりも前に宣言してね、と言われてしまう。
// fun generateContainer3<E>(vararg elements: E): = Containers(*elements)

fun main(vararg args: String) {
    val container = Containers("nasu", "kyuri", "gobou", "daikon")
    val container2 = generateContainer("seri", "nazuna", "gogyo", "hakobera")

    println(container)
    println(container2)
}
