global.alert = console.log;
global.fetch = require('node-fetch');
const userPool = require('./myUserPool');

require('./signin');