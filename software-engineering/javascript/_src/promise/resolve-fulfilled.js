// Promiseのresolveに対して、then()メソッドで後からコールバックを足した場合
const wait = ms => new Promise(resolve => setTimeout(resolve, ms));
wait(3000).then(() => console.log("3 seconds")).catch(() => console.log("error"));

// Promiseオブジェクトの引数のexecutorはresolve関数を引数にとるけど、そもそもexecutorを実行する段階でresolve関数を登録しないのはどうして？
// → 処理の前に予め全てのコールバックを登録する書き方になってしまうことを避けるため（コールバック地獄）

// thenメソッドの引数のonFulfilled関数をexecutorの引数として登録できるのはどうして？
// → おそらくPromiseオブジェクトがexecutorに渡した関数の参照を持っていて、resolveしたら渡すようにしているため