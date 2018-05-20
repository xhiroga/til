# Firebase
Googleが提供するMobile Backend as a Service。  


# Usage
## Realtime Database
node.js
```console
yarn add firebase
node

const firebase = require('firebase');
const config = require('./config.js');
firebase.initializeApp(config);
const ref = firebase.database().ref('user');
ref.set({name:'hiro'});
```

## Cloud Firestore
次世代版のRealtime Database。SQLのようにクエリが使用可能。