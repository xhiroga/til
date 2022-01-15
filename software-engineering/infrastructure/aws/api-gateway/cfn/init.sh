set -eu

S3_BUCKET_CFN=localstack-cfn
aws s3 mb ${S3_BUCKET_CFN}

aws cloudformation package \
    --template-file cfn/template.yml \
    --s3-bucket ${S3_BUCKET_CFN} \
    --s3-prefix "templates" \
    --output-template-file cfn/output-template.yml

aws cloudformation deploy \
    --stack-name localstack \
    --template-file /cfn/output-template.yml \
    --endpoint-url http://localstack:4581

