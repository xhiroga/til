<template>
  <v-layout column justify-center align-center>
    <v-card v-if="todos">
      <v-card-title>
        TODO一覧
        <v-spacer />
        <v-text-field
          v-model="search"
          append-icon="mdi-magnify"
          label="検索"
          sigle-line
        />
      </v-card-title>
      <!-- 新規追加 -->
      <v-btn fab dark small color="dark" class="mb-2" @click="add">
        <v-icon dark>
          mdi-plus
        </v-icon>
      </v-btn>
      <v-data-table>
        <!-- モーダル -->
        <template v-slot:top>
          <v-dialog v-model="dialog" max-width="500px">
            <v-card>
              <v-card-title>
                <span class="headline">{{ formTitle }}</span>
              </v-card-title>
              <v-card-text>
                <v-container>
                  <v-row>
                    <v-col cols="12">
                      <v-text-field v-model="todo.task" label="タスク" />
                    </v-col>
                  </v-row>
                </v-container>
              </v-card-text>
              <v-card-actions>
                <v-spacer />
                <v-btn @click="close">閉じる</v-btn>
                <v-btn v-if="isPersistedTodo" class="primary" @click="update"
                  >更新する
                </v-btn>
                <v-btn v-else class="primary" @click="create">追加する</v-btn>
                <v-spacer />
              </v-card-actions>
            </v-card>
          </v-dialog>
        </template>
        <!-- 編集と削除 -->
        <template v-slot:[`item.actions`]="{ item }">
          <v-icon small @click="edit(item)">
            mdi-pencil
          </v-icon>
          <v-icon small @click="remove(item)">
            mdi-delete
          </v-icon>
        </template>
      </v-data-table>
    </v-card>
  </v-layout>
</template>
<script>
import axios from "axios";

export default {
  async asyncData({ app }) {
    const response = await axios.get("http://localhost:3000/api/todos");
    return { todos: response.data };
  },
  data() {
    return {
      search: "",
      headers: [
        { text: "ID", value: "id" },
        { text: "タスク", value: "task" },
        { text: "操作", value: "actions" }
      ],
      todo: {}
    };
  },
  computed: {
    isPersistedTodo() {
      return !!this.todo.id;
    },
    formTitle() {
      return this.isPersistedTodo ? "TODO編集" : "TODO追加";
    }
  },
  methods: {
    add(todo) {
      this.todo = {};
      this.dialog = true;
    },
    async create() {
      await axios.post("/api/todos", this.todo).then(() => {
        this.$router.app.refresh();
      });
      this.close();
    },
    edit(todo) {
      this.todo = Object.assign({}, todo);
      this.dialog = true;
    },
    async update() {
      await axios.put("/api/todos/" + this.todo.id, this.todo).then(() => {
        this.$router.app.refresh();
      });
      this.close();
    },
    async remove(todo) {
      await axios.delete("/api/todos/" + todo.id, todo).then(() => {
        this.$router.app.refresh();
      });
    },
    close() {
      this.dialog = false;
      this.todo = {};
    }
  }
};
</script>
