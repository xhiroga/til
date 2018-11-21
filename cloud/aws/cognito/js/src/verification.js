require('./common')
const AmazonCognitoIdentity = require('amazon-cognito-identity-js');
const userPoolFactory = require('./userPoolFactory');
const userPool = userPoolFactory()

const userData = {
    Username : process.env.USERNAME,
    Pool : userPool
};

var cognitoUser = new AmazonCognitoIdentity.CognitoUser(userData);
cognitoUser.confirmRegistration(process.env.VERIFICATION_CODE, true, function(err, result) {
    if (err) {
        alert(err);
        return;
    }
    console.log('call result: ' + result);
});
