#!/bin/ash

CODE_REPO=https://git-codecommit.us-east-1.amazonaws.com/v1/repos/batch
CODE_DIR=/usr/local/test-batch

git clone --config credential.helper='!aws --region ap-northeast-1 codecommit credential-helper $@' --config credential.UseHttpPath=true $CODE_REPO $CODE_DIR

sh /usr/local/test-batch/run.sh