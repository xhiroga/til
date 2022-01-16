package com.example.todo

import TodoService
import com.fasterxml.jackson.databind.SerializationFeature
import com.example.todo.web.todos
import com.example.todo.factory.DatabaseFactory
import io.ktor.application.*
import io.ktor.features.*
import io.ktor.http.ContentType
import io.ktor.jackson.*
import io.ktor.response.*
import io.ktor.request.*
import io.ktor.routing.*
import org.flywaydb.core.Flyway.configure

fun main(args: Array<String>): Unit = io.ktor.server.tomcat.EngineMain.main(args)

@Suppress("unused") // Referenced in application.conf
@kotlin.jvm.JvmOverloads
fun Application.module(testing: Boolean = false) {
    install(ContentNegotiation) {
        jackson {
            configure(SerializationFeature.INDENT_OUTPUT, true)
        }
    }
    DatabaseFactory.init()

    val todoService = TodoService()
    install(Routing) {
        todos(todoService)
    }

    routing {
        get("/") {
            call.respondText("Hello, world!", ContentType.Text.Plain)
        }
    }
}

