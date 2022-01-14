# Environment Variables

# Usage
# デフォルトのプロファイルを変更する。プロファイルとリージョンで環境変数が別なので注意。
```bash
export AWS_PROFILE=test-user
export AWS_DEFAULT_PROFILE=test-user # CLIでは上記よりこちらが優先される。
export AWS_DEFAULT_REGION=us-east-1
```

# Tips
```bash
alias aws-test="export AWS_DEFAULT_PROFILE=test-user;export AWS_DEFAULT_REGION=us-east-1;echo -e '切り替え完了！\nAWS_DEFAULT_PROFILE=test-user\nAWS_DEFAULT_REGION=us-east-1'"
```