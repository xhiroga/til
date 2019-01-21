fun sum(vararg ints: Int): Int {
    return ints.reduce({ i1, i2 -> i1 + i2 })
}

fun main(args: Array<String>) {
    println(sum(1, 2, 3, 4, 5))

    val num = intArrayOf(2, 4, 6, 5, 3)
    println(sum(*num))

    // val (first, others) = *num // 分解宣言には使えない
}
