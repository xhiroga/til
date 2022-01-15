package com.example.todo.web

import TodoService
import com.example.todo.model.NewTodo
import io.ktor.application.*
import io.ktor.http.*
import io.ktor.request.*
import io.ktor.response.*
import io.ktor.routing.*
import java.lang.IllegalStateException

fun Route.todos(todoService: TodoService){
    route("/todos"){
        get("/"){
            call.respond(todoService.getAllTodos())
        }

        get("/{id}"){
            val id = call.parameters["id"]?.toInt() ?: throw IllegalStateException("Must To iD")
            val todo = todoService.getTodo(id)
            if(todo == null) call.respond(HttpStatusCode.NotFound)
            else call.respond(todo)
        }

        post("/"){
            val newTodo = call.receive<NewTodo>()
            call.respond(HttpStatusCode.Created, todoService.addTodo(newTodo))
        }

        put("/{id}"){
            val todo = call.receive<NewTodo>()
            val updated = todoService.updateTodo(todo)
            if(updated == null) call.respond(HttpStatusCode.NotFound)
            else call.respond(HttpStatusCode.OK,updated)
        }

        delete("/{id}"){
            val id = call.parameters["id"]?.toInt() ?: throw IllegalStateException("Must To id")
            val removed = todoService.deleteTodo(id)
            if (removed) call.respond(HttpStatusCode.OK)
            else call.respond(HttpStatusCode.NotFound)
        }
    }
}