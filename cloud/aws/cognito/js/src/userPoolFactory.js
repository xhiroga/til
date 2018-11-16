const AmazonCognitoIdentity = require('amazon-cognito-identity-js');
const CognitoUserPool = AmazonCognitoIdentity.CognitoUserPool;

const poolData = {
    UserPoolId : process.env.USER_POOL_ID,
    ClientId : process.env.CLIENT_ID
};

module.exports = () => new CognitoUserPool(poolData);
