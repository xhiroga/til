# AWS Lambda Custom Runtime AWS CLI by CFn

```shell script
docker run --rm -it -v "$(pwd)/src/layer:/opt" hiroga/amazonlinux-pip pip install -r requirements.txt -t ./packages

APPLICATION=custom-runtime-awscli-cfn
S3_BUCKET_CFN=cc.hiroga.cfn

aws cloudformation package \
    --template-file template.yml \
    --output-template-file template-output.yml \
    --s3-bucket ${S3_BUCKET_CFN} \
    --s3-prefix "${APPLICATION}/artifacts"

aws cloudformation deploy \
    --template-file template-output.yml \
    --s3-bucket ${S3_BUCKET_CFN} \
    --s3-prefix "${APPLICATION}/deploy" \
    --stack-name "${APPLICATION}" \
    --capabilities CAPABILITY_IAM CAPABILITY_AUTO_EXPAND
```
