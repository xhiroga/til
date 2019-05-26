bucket_name=$1
bucket_uri="s3://${1}"
aws s3 sync public/ $bucket_uri --acl public-read

region=$(aws configure get region)
echo "http://${1}.s3-website-${region}.amazonaws.com"