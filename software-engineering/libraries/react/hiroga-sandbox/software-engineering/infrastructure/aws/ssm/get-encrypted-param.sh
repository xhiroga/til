#!/usr/bin/env sh

set -euxo pipefail

NAME=${1}

aws ssm get-parameter \
    --name ${NAME} \
    --with-decryption \
    --query Parameter.Value --output text
