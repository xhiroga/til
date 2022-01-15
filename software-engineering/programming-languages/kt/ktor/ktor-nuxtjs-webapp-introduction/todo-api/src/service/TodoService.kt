import com.example.todo.model.NewTodo
import com.example.todo.model.Todo
import com.example.todo.model.Todos
import com.example.todo.factory.DatabaseFactory.dbQuery
import org.jetbrains.exposed.sql.*

class TodoService {
    suspend fun getAllTodos(): List<Todo> = dbQuery {
        Todos.selectAll().map { convertTodo(it) }
    }
    suspend fun getTodo(id: Int): Todo? = dbQuery {
        Todos.select {
            (Todos.id eq id)
        }.mapNotNull { convertTodo(it) }
            .singleOrNull()
    }
    suspend fun addTodo(todo: NewTodo): Todo {
        var key = 0
        dbQuery {
            key = (Todos.insert {
                it[task] = todo.task
            } get Todos.id)
        }
        return getTodo(key)!!
    }
    suspend fun updateTodo(todo: NewTodo): Todo? {
        val id = todo.id
        return if (id == null) {
            addTodo(todo)
        } else {
            dbQuery {
                Todos.update({ Todos.id eq id }) {
                    it[task] = todo.task
                }
            }
            getTodo(id)
        }
    }

    suspend fun deleteTodo(id: Int): Boolean {
        return dbQuery {
            Todos.deleteWhere { Todos.id eq id } > 0
        }
    }

    private fun convertTodo(row: ResultRow): Todo =
        Todo(
            id = row[Todos.id],
            task = row[Todos.task]
        )
}