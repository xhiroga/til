// 実行するとjsonを返す関数
// ステータスならステータスを返す関数。更新とかも。

import data from './LibraryList.json';
// jsonなので拡張子を忘れずに。
console.log('data passed!');
console.log(data);

export default () => data;
