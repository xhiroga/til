immutable = require('immutable')
List = immutable.List

/*
そもそもの前提として、JavaScriptの組み込み型にList型はない。
Listはimmutable.jsで初めて提供されている型である。
*/

array = new Array()
array.push(1, 2, 3)
console.log(array)
// 配列の要素数（プロパティ）
console.log('📏 length', array.length)
// setの要素数
console.log('🏋️‍♀️ size', array.size)

imList = List.of(1, 2, 3, 4, 5)
console.log(imList)
// 提供されていない
console.log('📏 length', imList.length)
// Listのサイズを返す
console.log('🏋️‍♀️ size', imList.size)

//////////

console.log('おまけ')
array.length = 10000
console.log('lengthを上書きしたら? ', array) // [ 1, 2, 3, <9997 empty items>, size: 10000 ]
console.log('lengthの値は? ', array.length) // 10000