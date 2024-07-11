# Open ID federation in Federated Identities

To understand Federated Identity - Facebook connection.

## How to do

```shell
FacebookAppId=${FacebookAppId}
aws cloudformation deploy \
    --template-file "template.yml" \
    --stack-name "federated-identity-open-id" \
    --capabilities CAPABILITY_IAM CAPABILITY_AUTO_EXPAND CAPABILITY_NAMED_IAM \
    --parameter-overrides "FacebookAppId=${FacebookAppId}" \
    --tags date="$(date '+%Y%m%d%H%M%S')" \
    --region ap-northeast-1
```

```shell
gem install facebook-cli
facebook-cli config --appid=${FacebookAppId} --appsecret=${FacebookSecret}
facebook-cli login
open ~/.facebook-clirc
```

```shell
AccessToken=${AccessToken}
FederatedIdentityId=${FederatedIdentityId}
aws cognito-identity get-id \
    --identity-pool-id ${FederatedIdentityId} \
    --logins "{\"graph.facebook.com\":\"${AccessToken}\"}" \
    --region ap-northeast-1

# {
#     "IdentityId": "ap-northeast-1:960db267-2e5e-4135-83b1-d01a7a0bc9c2"
# }

IdentityId=${IdentityId}
aws cognito-identity describe-identity \
    --identity-id ${IdentityId} \
    --region ap-northeast-1

# {
#     "IdentityId": "ap-northeast-1:960db267-2e5e-4135-83b1-d01a7a0bc9c2",
#     "Logins": [
#         "graph.facebook.com"
#     ],
#     "CreationDate": 1581242169.197,
#     "LastModifiedDate": 1581242169.217
# }
```
