fun main(args: Array<String>) {

    // Error:(3, 15) Kotlin: 'if' must have both main and 'else' branches if used as an expression
    // val res = if(true) "TRUE" // elseの省略はだめ
    // val res: String? = if(true) "TRUE" // ヌル許可にすればいいという話ではない
    // val res = if(true) "TRUE" else; // こういう省略もだめ

    val res = if (true) "TRUE" else "FALSE"
    println(res)

    // ifは式である、というのは println(if(1==1)) ということではなく...
    println((1 == 1)) // こういうこと
}