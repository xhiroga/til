require('./common')
const AmazonCognitoIdentity = require('amazon-cognito-identity-js');
const userPoolFactory = require('./myUserPool');
const userPool = userPoolFactory()

const userData = {
    Username : 'hiroaki',
    Pool : userPool
};

var cognitoUser = new AmazonCognitoIdentity.CognitoUser(userData);
cognitoUser.confirmRegistration('026795', true, function(err, result) {
    if (err) {
        alert(err);
        return;
    }
    console.log('call result: ' + result);
});
