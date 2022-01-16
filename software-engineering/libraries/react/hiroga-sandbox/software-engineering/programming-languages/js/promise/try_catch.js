// async/awaitは、本質的には呼び出し側の記述をthenの内側で実行しているに過ぎないと思われる。
// そのため、try-catchをするにしても、対象の非同期関数をawaitで呼び出さないと、catchが非同期処理の対象にならない。

const unhandled = () => {
    try {
        return new Promise((resovle, reject) => {
            reject('this promise will be rejected every time🤡')
        })
    } catch (err) {
        console.log('ERROR', err)
    }
}
unhandled()
/*
(node:89978) UnhandledPromiseRejectionWarning: this promise will be rejected every time🤡
(node:89978) UnhandledPromiseRejectionWarning: Unhandled promise rejection. This error originated either by throwing inside of an async function without a catch block, or by rejecting a promise which was not handled with .catch(). (rejection id: 1)
(node:89978) [DEP0018] DeprecationWarning: Unhandled promise rejections are deprecated. In the future, promise rejections that are not handled will terminate the Node.js process with a non-zero exit code.
*/

const handled = async () => {
    try {
        return await new Promise((resovle, reject) => {
            reject('this promise will be rejected every time🤡')
        })
    } catch (err) {
        console.log('ERROR', err)
    }
}

handled()
// ERROR this promise will be rejected every time🤡