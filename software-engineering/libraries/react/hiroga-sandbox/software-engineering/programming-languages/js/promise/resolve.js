// Promiseを返す引数が、もしも呼び出し時点でresolveを決めていたら？

// Promiseコンストラクタの引数のコールバック関数の引数をあらかじめ決めておこう、ということになる。
// Javaと違って関数をどこでも宣言できるので、その時点での変数が変数として使えそうに見えてしまうせい。実際には↑↑はできない。

const おみくじ = (resolve1) => {
    num = Math.random()
    if (num > 0.5) {
        結果 = '吉 😄'
    } else {
        結果 = '凶 😈'
    }

    // この引数resolveをあらかじめ固定することはできない(やるとしたら、bindなどを使うことになる)
    return new Promise((resolve, reject) => {
        resolve(結果)
    })
}

おみくじ().then((結果) => console.log(結果))