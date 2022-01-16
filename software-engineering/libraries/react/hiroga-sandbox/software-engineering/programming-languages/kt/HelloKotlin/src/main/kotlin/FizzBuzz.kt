
fun main(args: Array<String>){
//    for (i in 1..100){
//        println(if (i % 3 == 0 && i % 5 == 0) "FizzBuzz" else if (i % 3 == 0) "Fizz" else if (i % 5 == 0) "Buzz" else i.toString())
//    }
    for (i in 1..100){
        val num = when {
            i % 15 == 0 -> "FizzBuzz"
            i % 3 == 0 -> "Fizz"
            i % 5 == 0 -> "Buzz"
            else -> "$i"
        }
        println(num)
    }
}