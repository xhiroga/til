# AWS CLI

```shell
aws configure ＃認証情報の設定
aws s3 ls ＃認証情報が設定されているかを確かめるのに便利
```

## Aliases

~/.aws/cli/aliasに{alias}={command}の形式でaws cliのコマンドを登録できる。  

### Aliases usage

```shell
cat << EOS > ~/.aws/cli/alias
[toplevel]

whoami = sts get-caller-identity
EOS
```

## Environment variables

### Environment variables usage

```shell
export AWS_PROFILE=test-user
export AWS_DEFAULT_PROFILE=test-user # CLIでは上記よりこちらが優先される。
export AWS_DEFAULT_REGION=us-east-1
```

## References

- [awslabs/awscli-aliases](https://github.com/awslabs/awscli-aliases)
