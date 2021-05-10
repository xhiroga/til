require('./common')
const AmazonCognitoIdentity = require('amazon-cognito-identity-js');
const userPoolFactory = require('./userPoolFactory');
const userPool = userPoolFactory()

var attributeList = [];

var dataEmail = {
    Name : 'email',
    Value : process.env.EMAIL
};
var dataPhoneNumber = {
    Name : 'phone_number',
    Value : process.env.PHONE
};
var attributeEmail = new AmazonCognitoIdentity.CognitoUserAttribute(dataEmail);
var attributePhoneNumber = new AmazonCognitoIdentity.CognitoUserAttribute(dataPhoneNumber);

attributeList.push(attributeEmail);
attributeList.push(attributePhoneNumber);

// コンソールに反映されるまで15秒くらいかかる
userPool.signUp(process.env.USERNAME, process.env.PASSWORD, attributeList, null, function(err, result){
    if (err) {
        alert(err);
        return;
    }
    cognitoUser = result.user;
    console.log('user name is ' + cognitoUser.getUsername()); // user name is hiroaki
});