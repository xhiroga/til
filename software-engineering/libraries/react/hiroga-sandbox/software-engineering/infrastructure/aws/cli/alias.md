# Aliases
~/.aws/cli/aliasに{alias}={command}の形式でaws cliのコマンドを登録できる。  

# Usage
```bash
cat << EOS > ~/.aws/cli/alias
[toplevel]

whoami = sts get-caller-identity
EOS
```

# Reference
https://github.com/awslabs/awscli-aliases