// Promiseを上手に使う

const axios = require('axios');


// 前提として、axios.get()はPromiseオブジェクトを返すのか？
var res = axios.get('https://google.com');
console.log(Object.getPrototypeOf(res)); // Promise {}


// 1. 結果を取得したらコールバックする関数が作りたい
const getCallback = (url, callback) => {
    // axios.get()の内側では、Promise()をすぐにnewして返している。
    // Promiseコンストラクタはexecutor関数をすぐに実行し、次にPromiseオブジェクトを返す。
    // executor関数はロジックに応じてresolve()またはreject()を呼び出す。ただし、この時点ではresolve()は登録されていない。
    const pro1 = axios.get(url)

    // then()メソッドは、Promiseの内部状態に応じて呼び出されるresolve関数を登録し、Promise自身を返す。
    const pro2 = pro1.then((res) => {
        callback(res.status)
    })
}
getCallback('https://google.com', (val) => console.log('getCallback: ', val)) // getCallback:  200


// 2.結果を取得してから返す(ように見える)関数を作りたい。
// → 即ち、resolveに渡された値を元の同期処理側で受け取る(ように見える)演算子が必要になる。
const getResolve = async url => {
    // axios.get()が生成するPromiseオブジェクトは、resolve()の引数にresponseを与えている。
    // awaitは、引数をそのまま返すような関数を暗黙的にresolveに渡していると推測される。
    const res = await axios(url)
    return res.status + "000" // getResolve()自体はPromiseを返すが、async関数になったことでreturnされた値をresolveに渡せる。
}
getResolve('https://google.com').then(stat => console.log('getResolve: ', stat)) // getResolve:  200000


// 結局のところ、Promiseオブジェクトから返された値を同期処理側に引き戻すことはできない(と思う)