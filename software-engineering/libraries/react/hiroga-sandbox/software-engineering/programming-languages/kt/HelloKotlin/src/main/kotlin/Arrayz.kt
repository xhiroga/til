fun main(args: Array<String>) {
    // 配列およびコレクション型にはそれぞれファクトリメソッドが用意されている

    val intp = intArrayOf(1, 1, 2, 4, 8) // プリミティブ型の配列に相当する
    println(intp)

    val inta = arrayOf(1, 2, 3)
    println(inta)
    println(inta[0])

    // Collection
    // val listp = intListOf(2,4,6,8) // これは存在しない

    val listz = listOf(1, 3, 5, 7)
    println(listz is Collection<Int>)
    println(listz[0])

    val mapz = mapOf("one" to 1)
    // println(mapz is Collection<String, Int>) // Collectionの型パラメータは引数を一つしか受け取れないので、これは誤り。
    println(mapz["one"])

    val setz = setOf(1, 2, 3)
    println(setz is Collection<Int>)
    println(setz.first())
}