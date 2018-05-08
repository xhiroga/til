var express = require('express');
var app = express();

// ハンドラー
app.get('/', (req, res) => {res.send('Hello, wolrd!')});

// サーバー起動
var server = app.listen(3000, () => {
    console.log('Server Listening!!');
});

// Reference
// https://gist.github.com/mitsuruog/fc48397a8e80f051a145