require('./common')

const getIdToken = require('./getIdToken');
const getCognitoIdentityCredentials = require('./getCognitoIdentityCredentials')

getIdToken(idToken => getCognitoIdentityCredentials(idToken, console.log))