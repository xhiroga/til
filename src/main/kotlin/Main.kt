package todolist

// MEMO: 自作クラスがimportできなかったら、それはpackageが同じかどうかを疑うといいかも

import io.ktor.application.*
import io.ktor.http.*
import io.ktor.response.*
import io.ktor.routing.*
import io.ktor.server.engine.*
import io.ktor.server.netty.*
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.kotlin.registerKotlinModule
import java.lang.Exception

fun main(args: Array<String>) {
    embeddedServer(Netty, 8080) {
        routing {
            get("/") {
                call.respondText("My Example Blog", ContentType.Text.Html)
            }
            get("/tasks") {
                try {
                    println(call.request)

//               TODO: なぜかObjectMapper()のインスタンスを作るとレスポンスが返らなくなる。
//               val objectMapper = ObjectMapper().registerKotlinModule()

                    val tasks = listOf(
                        Task(1, "みかん販売の看板を作成する", false),
                        Task(2, "トイレットペーパーを購入する", true)
                    )
                    // call.respondがtransformしてくれるのはdata class単体に限るらしい？
                    call.respondText(
                        tasks.toString()
                    )
                } catch (error: Exception) {
                    println(error)
                }
            }
        }
    }.start(wait = true)
}

// See: https://ktor.io/quickstart/quickstart/gradle.html

// TODO: Separate applicatoin module from main function
// TODO: extract configuration
