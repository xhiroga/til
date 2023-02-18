bucket_name=$1
bucket_uri="s3://${bucket_name}"

aws s3 mb $bucket_uri

aws s3 website \
    --index-document index.html \
    --error-document error.html \
    $bucket_uri