package com.example.todo.model

import org.jetbrains.exposed.sql.Column
import org.jetbrains.exposed.sql.Table

object Todos: Table() {
    val id: Column<Int> = integer("id").autoIncrement().primaryKey()
    val task: Column<String> = varchar("task", 4000)
}
data class Todo (
    val id: Int,
    val task: String
)
data class NewTodo (
    val id: Int?,
    val task: String
)