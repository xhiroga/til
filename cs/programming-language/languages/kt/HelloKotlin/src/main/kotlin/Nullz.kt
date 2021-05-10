fun main(args: Array<String>) {
    isNull("STRING!")
    elvis("STRING!")

    isNull(null)
    elvis(null)
}

fun isNull(str: String?) {
    // ifの内側がオブジェクトそのものを取ることができない代わりに?エルビス演算子がある。
    // if(str) これは不可能
    // でも null == null は true
    val res = if (str == null) "default" else str
    println(res)
}

fun elvis(str: String?) {
    val res = str ?: "default"
    println(res)
}