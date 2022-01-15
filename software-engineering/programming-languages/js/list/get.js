const { List } = require('immutable')

imList = List.of(1, 2, 3, 4, 5)
console.log('👌 by index: ', imList[3])
console.log('👌 by index: ', imList.get(3))