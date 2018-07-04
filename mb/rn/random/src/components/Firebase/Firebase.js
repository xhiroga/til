import React, { Component } from 'react';
import {
    ActivityIndicator,
    Button,
    KeyboardAvoidingView,
    ListView,
    Text,
    TextInput,
    TouchableHighlight,
    TouchableOpacity,
    View
} from 'react-native';

import firebase from 'firebase';
// 公式SDKとTHREE.jsを併用できない原因不明のバグあり(2018/05/20)
// https://github.com/firebase/firebase-js-sdk/issues/538

import { apiKey, authDomain, databaseURL, projectId, storageBucket, messagingSenderId } from 'react-native-dotenv'
// .envファイルはkey=value形式(valueを""で囲ってはいけない)のため注意。
// しかも、.envを書き換えたら`rm -rf $TMPDIR/react-*`でキャッシュファイルの削除が必要。
// .babelrcとyarn add path わずれずに


import styles from './styles';

// Initialize Firebase
const firebaseConfig = {
    apiKey: apiKey,
    authDomain: authDomain,
    databaseURL: databaseURL,
    projectId: projectId,
    storageBucket: storageBucket,
    messagingSenderId: messagingSenderId,
};
console.log(firebaseConfig);
firebase.initializeApp(firebaseConfig);

const ref = firebase.database().ref('todo');
console.log("database refs");
console.log(ref);

export default class Firebase extends Component {

    constructor(props) {
        super(props);

        const dataSource = new ListView.DataSource({
            rowHasChanged: (r1, r2) => r1 !== r2
        })

        this.todos = {}
        this.state = {
            dataSource: dataSource.cloneWithRows(this.todos),
            numofTodos: Object.keys(this.todos).length,
            newTodo: '',
        }

        ref.on('child_added', snapshot => {
            this.todos[snapshot.key] = snapshot.val()
            this.setState({
                dataSource: this.state.dataSource.cloneWithRows(this.todos),
                numofTodos: Object.keys(this.todos).length,
            })
            console.log("ref.on('child_added'... called.");
            console.log(this.state);
        })
        ref.on('child_changed', snapshot => {
            this.todos = {
                ...this.todos
            }
            this.todos[snapshot.key] = snapshot.val()
            this.setState({
                dataSource: this.state.dataSource.cloneWithRows(this.todos),
            })
            console.log("ref.on('child_changed'... called.");
            console.log(this.state);
        })
        ref.on('child_removed', snapshot => {
            delete this.todos[snapshot.key]
            // 指定したオブジェクトを削除する(delete演算子)
            this.setState({
                dataSource: this.state.dataSource.cloneWithRows(this.todos),
                numofTodos: Object.keys(this.todos).length,
            })
            console.log("ref.on('child_removed'... called.");
            console.log(this.state);
        })
    }

    componentDidUpdate() {
        console.log(this.todos)
    }

    _addTodo(event) {
        console.log("_addTodo called. This is state");
        console.log(this.state);
        ref.push().set({
            todo: this.state.newTodo,
            isDone: false,
            createdAt: (new Date()).toISOString(),
        })
        this.setState({
            newTodo: '',
        })
    }

    _setDone(rowId, newValue) {
        ref.child(rowId).update({
            isDone: newValue,
        })
    }

    _deleteTodo(rowId) {
        ref.child(rowId).set(null)
    }

    _onChangeText(text) {
        this.setState({
            newTodo: text,
        })
    }

    _renderRow(rowData, sectionId, rowId) {
        console.log("_rednerRow called.");
        console.log(rowData);
        return (
            <View style={styles.cell}>
                <View style={styles.checkbox}>
                    <Button
                        title={rowData.todo}
                        checked={rowData.isDone}
                        onPress={() => this._setDone(rowId, !rowData.isDone)}
                    />
                </View>
                <TouchableOpacity
                    onPress={() => this._deleteTodo(rowId)}
                    style={styles.deleteButton}
                >
                    <Text style={styles.deleteLabel}>Del</Text>
                </TouchableOpacity>
            </View>
        )
    }

    render() {
        return (
            <KeyboardAvoidingView
                behavior='padding'
                style={styles.container}
            >
                {0 < this.state.numofTodos
                    ? <ListView
                        dataSource={this.state.dataSource}
                        renderRow={this._renderRow.bind(this)}
                        enableEmptySections={true}
                        style={styles.todos}
                    />
                    : <ActivityIndicator
                        animating={true}
                        size="large"
                        style={styles.todos}
                    />
                }

                <View style={styles.input}>
                    <TextInput
                        value={this.state.newTodo}
                        placeholder="Write todo here!"
                        onChangeText={event => this._onChangeText(event)}
                        onSubmitEditing={event => this._addTodo(event)}
                        blurOnSubmit={true}
                        keyboardType="default"
                        returnKeyType="done"
                        style={styles.textinput}
                    />
                </View>
            </KeyboardAvoidingView>

        )
    }
}

// Reference
// https://github.com/januswel/fb-rn-todo/