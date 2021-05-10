#!/usr/bin/env sh

set -euxo pipefail

aws ses send-email \
  --from "hiroga1030@gmail.com" \
  --destination "ToAddresses=hiroga1030@gmail.com,CcAddresses=hiroga1030@gmail.com" \
  --message "Subject={Data=from aws cli,Charset=utf8},Body={Text={Data=sent from aws cli,Charset=utf8},Html={Data=,Charset=utf8}}" \
  --profile "${PROFILE}"

### Reference
# https://docs.aws.amazon.com/cli/latest/reference/ses/send-email.html
