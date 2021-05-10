require('dotenv').config()

const getIdToken = require('./getIdToken')
const getCognitoIdentityCredentials = require('./getCognitoIdentityCredentials')
setPostman = (credentials) => {
    pm.environment.set("accessKey", credentials.accessKey)
    pm.environment.set("secretKey", credentials.secretKey)
    pm.environment.set("sessionToken", credentials.sessionToken)
}

getIdToken(idToken => getCognitoIdentityCredentials(idToken, console.log))
pm.environment.set("region", process.env.REGION)