const axios = require('axios');

const _ignoreErr = (err) => {
    console.log('ERROR:', err)
    // どんなerrを受け取ったとしてもresolveして、エラーを握りつぶす。
    return new Promise((resolve, reject) => { resolve() })
}
const getNotExist = async () => {
    try {
        axios.get('http://not.exist.com').catch(_ignoreErr)
    } catch (err) {
        // 上記 _ignoreErr で握りつぶしているので、発火しないはずでは...
        console.log('this message will not shown:', err)
    }
}

getNotExist()