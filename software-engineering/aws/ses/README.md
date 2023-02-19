# AWS SES

```shell
#!/usr/bin/env sh

set -euxo pipefail

aws ses send-email \
  --from "${EMAIL}" \
  --destination "ToAddresses=${EMAIL},CcAddresses=${EMAIL}" \
  --message "Subject={Data=from aws cli,Charset=utf8},Body={Text={Data=sent from aws cli,Charset=utf8},Html={Data=,Charset=utf8}}" \
  --profile "${PROFILE}"
```

## References

- [send](https://docs.aws.amazon.com/cli/latest/reference/ses/send-email.html)
