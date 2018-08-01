/*
    IAMユーザーに適切なPermissionを設定する必要がある
    Action:"s3:*"がAllowになっている必要がある
*/

var AWS = require('aws-sdk');
var fs = require('fs');

// Credentialの読み込み方法は何種類かある
// https://docs.aws.amazon.com/sdk-for-javascript/v2/developer-guide/setting-credentials-node.html

AWS.config.loadFromPath('./config.json');

// jsonファイルに記載しない場合は、update()で 情報を追加できる
// AWS.config.update({ region: 'リージョン名' });

// S3()コンストラクターの引数にoptionを設定する方法もある
var s3 = new AWS.S3();
var params = {
    Bucket: "test.in.case.hiroga",
    Key: "Lena.jpg"
};
var v = fs.readFileSync("./Lena.jpg");
params.Body = v;
s3.putObject(params, function (err, data) {
    if (err) console.log(err, err.stack);
    else console.log(data);
});