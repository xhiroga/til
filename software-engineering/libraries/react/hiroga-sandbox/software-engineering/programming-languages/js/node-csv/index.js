const csv = require('csv');

input = [
    ["1", "2", "3"],
    ["a", "b", "c"]
]

// csvを文字列に変換する
csv.stringify(input, function (err, output) {
    console.log(output)
})

