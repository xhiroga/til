package todolist

class TaskRepository{
    private val tasks: MutableList<Task> = mutableListOf()

    private val maxId: Long
        // メンバーリファレンスを使うと引数でインスタンスを受け取る箇所の記述を省ける。
        get() = tasks.map(Task::id).max() ?: 0

    // リスト本体ではなくコピーを返すためのtoList() これはメソッドではなく拡張関数らしい...が、それって違いを解説する意味があるのか？
    fun findAll(): List<Task> = tasks.toList()

    fun create(content: String): Task{
        val id = maxId + 1
        val task = Task(id, content, false)
        tasks += task
        return task
    }

}