require('dotenv').config()

// Modules, e.g. Webpack:
var AmazonCognitoIdentity = require('amazon-cognito-identity-js');
var CognitoUserPool = AmazonCognitoIdentity.CognitoUserPool;

var poolData = {
    UserPoolId : process.env.USER_POOL_ID,
    ClientId : process.env.CLIENT_ID
};

module.exports = () => new CognitoUserPool(poolData);