const AmazonCognitoIdentity = require('amazon-cognito-identity-js');
const userPoolFactory = require('./userPoolFactory');
const userPool = userPoolFactory()

var authenticationData = {
    Username : process.env.USER,
    Password : process.env.PASSWORD,
};
const userData = {
    Username : process.env.USER,
    Pool : userPool
};

var authenticationDetails = new AmazonCognitoIdentity.AuthenticationDetails(authenticationData);
var cognitoUser = new AmazonCognitoIdentity.CognitoUser(userData);

module.exports = (callback) => {
    cognitoUser.authenticateUser(authenticationDetails, {
        onSuccess: function (result) {
            var idToken = result.getIdToken().getJwtToken();
            console.log({idToken: idToken})
            callback(idToken)
        },
    
        onFailure: function(err) {
            console.log(err);
        },
        mfaRequired: function(codeDeliveryDetails) {
            var verificationCode = prompt('Please input verification code' ,'');
            cognitoUser.sendMFACode(verificationCode, this);
        }
    });
}