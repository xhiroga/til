fun showStrings(vararg strs: String) {
    for (str in strs) {
        println(str)
    }
}

fun showFirst(str: String) {
    println(str)
}

fun showNumbers(vararg ints: Int) {
    for (int in ints) {
        println(int)
    }
}

// mainでvarargも使える！これはタイプ数の削減に繋がって楽チンかも。
fun main(vararg args: String) {
    val strs: Array<String> = arrayOf("ringo", "gorilla", "rappa", "panty")
    showStrings(*strs)
//    showFirst(*strs) // 引取先がvarargキーワードを使っていない場合は使用できない。

//    val ints: Array<Int> = arrayOf(1,2,3,4,5) // これはスプレッド展開ができない...ざっけんな！！！！
    val ints: IntArray = intArrayOf(1, 2, 3, 4, 5)
    showNumbers(*ints)
}