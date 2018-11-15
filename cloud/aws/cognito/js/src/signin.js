require('./common')
const AmazonCognitoIdentity = require('amazon-cognito-identity-js');
const userPoolFactory = require('./myUserPool');
const userPool = userPoolFactory()

var authenticationData = {
    Username : 'hiroaki',
    Password : process.env.PASSWORD,
};
const userData = {
    Username : 'hiroaki',
    Pool : userPool
};

var authenticationDetails = new AmazonCognitoIdentity.AuthenticationDetails(authenticationData);
var cognitoUser = new AmazonCognitoIdentity.CognitoUser(userData);

cognitoUser.authenticateUser(authenticationDetails, {
    onSuccess: function (result) {
        var accessToken = result.getAccessToken().getJwtToken();
        console.log('accessToken: ',accessToken)
    },

    onFailure: function(err) {
        alert(err);
    },
    mfaRequired: function(codeDeliveryDetails) {
        var verificationCode = prompt('Please input verification code' ,'');
        cognitoUser.sendMFACode(verificationCode, this);
    }
});