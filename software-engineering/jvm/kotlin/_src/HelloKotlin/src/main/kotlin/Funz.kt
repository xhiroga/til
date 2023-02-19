// 関数の作り方には、大きく分けて関数宣言と関数リテラルがある
// また、関数リテラル（関数を作るための文法）にはラムダ式と匿名関数がある

fun main(args: Array<String>) {
    val got = double(123)
    println(got)

    val funT = ::triple
    println(funT(8))

    doFun(5, ::double)

    // {} の意味合いが関数宣言とラムダ式で異なるので注意
    // 1. function declaration なら関数本体の処理のみを示す
    // 2. lambda expression なら引数と返り値

    val lam = { n: Int, m: Int ->
        val p = n + m
        p
    }

    // Lambda深掘り
    // 最後に評価された値が返る
    // ただし early returnも可能
    // 返り値の型を宣言するのを省略できる
    val lam2 = lambda@{ n: Int, m: Int ->
        if (m == 0) {
            "not devided"
            return@lambda
        }
        n / m
    }

    println(max(1, 2, 3, 4, 5))

    val a = arrayOf(1, 2, 3)
    println(a)
}

// 式が1行なら{}で囲わずに = で宣言できる。ということは...
fun double(n: Int): Int = n * 2

fun triple(n: Int): Int = n * 3

fun hello(): Unit = println("hello world")
// Unitは省略できるので...
fun hello2() = println("hello")


fun doFun(n: Int, f: (Int) -> Int): Unit {
    val m = f(n)
    println(m)
}

// fun quad(n: Int): Int = println( n * 4 ); n*4 // これはできない。
// Lambda式では最後に評価された値が返却されるが、関数宣言では最後に評価された値が返る上にそもそも一つしか式を使えない

tailrec fun sum(ints: List<Int>, acc: Int): Int =
    if (ints.isEmpty()) acc
    else sum(ints.drop(1), acc + ints.first())

fun max(vararg ints: Int): Int {
    // 可変長引数の元々の定義がIntの場合はIntArrayになる（プリミティブ型に対応するものたち）
    var n = 0
    for (i in ints) {
        n = n + i
    }
    return n
}
