const amplify = require('aws-amplify');
global.fetch = require("node-fetch");
const Amplify = amplify.default
const Auth = amplify.Auth
const aws_exports = require('./aws-exports.withoutverification'); // https://aws-amplify.github.io/docs/js/authentication#manual-setup

username = `${Math.floor(Math.random() * 1000000)}`
email = `hiroga+${Math.floor(Math.random() * 1000000)}@example.com`
password = 'MyCoolP@ssw0rd1!'
console.log(username, email, password)

Amplify.configure(aws_exports);

const signUp = (username, email = null, password) => {
    attributes = {}
    if (email != null) attributes.email = email
    Auth.signUp({
        username,
        password,
        attributes
    })
        .then((data) => {
            console.log(data)
        }).catch((err) => {
            console.log(err)
        })
}

const confirmSignUp = (username, code) => {
    Auth.confirmSignUp(username, code, {
        // Optional. Force user confirmation irrespective of existing alias. By default set to True.
        forceAliasCreation: true
    }).then(data => console.log(data))
        .catch(err => console.log(err));

}

const signIn = (username, password) => {
    Auth.signIn(username, password)
        .then(data => {
            console.log(data)
        })
        .catch(err => {
            console.log(err)
        })
}
