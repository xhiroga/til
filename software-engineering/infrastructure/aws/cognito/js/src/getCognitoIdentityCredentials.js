const AWS = require('aws-sdk');

module.exports = (idJwtToken, callback) => {
    const userPoolUri = `cognito-idp.${process.env.REGION}.amazonaws.com/${process.env.USER_POOL_ID}`
    var logins = {}
    logins[userPoolUri] = idJwtToken
    config = AWS.config
    config.region = process.env.REGION
    config.credentials = new AWS.CognitoIdentityCredentials({
        IdentityPoolId: process.env.ID_POOL_ID,
        Logins: logins
    })
    config.credentials.get(function(err){
        if (err) {
            console.log('ERROR: ',err);
        }
        callback({
            accessKey: config.credentials.accessKeyId,
            secretKey: config.credentials.secretAccessKey,
            sessionToken: config.credentials.sessionToken
        })
    })
}

/*
    Refernces:
    https://docs.aws.amazon.com/ja_jp/cognito/latest/developerguide/tutorial-integrating-user-pools-javascript.html#tutorial-integrating-user-pools-getting-aws-credentials-javascript
    https://gist.github.com/eiKatou/25af74b2e1f62fb035106dc0d11264f2
*/