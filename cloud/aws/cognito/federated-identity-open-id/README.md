# Try to Federated Identity Open ID federation

To understand Federated Identity - Facebook connection.

## How to do

```sh
FacebookAppId=${FacebookAppId}
aws cloudformation deploy \
    --template-file "template.yml" \
    --stack-name "federated-identity-open-id" \
    --capabilities CAPABILITY_IAM CAPABILITY_AUTO_EXPAND CAPABILITY_NAMED_IAM \
    --parameter-overrides "FacebookAppId=${FacebookAppId}" \
    --tags date="$(date '+%Y%m%d%H%M%S')" \
    --region ap-northeast-1
```

```sh
gem install facebook-cli
facebook-cli config --appid=${FacebookAppId} --appsecret=${FacebookSecret}
facebook-cli login
open ~/.facebook-clirc
```
