# Ansible

## 注意事項

筆者は自宅サーバー管理のためにのみAnsibleを利用しています。ビジネス上のユースケースにマッチするスタイルガイドに鳴っていない可能性がありますが、ご了承下さい。

## Content Organization

[ベストプラクティス — Ansible Documentation](https://docs.ansible.com/ansible/2.9_ja/user_guide/playbooks_best_practices.html)([英](https://docs.ansible.com/ansible/2.8/user_guide/playbooks_best_practices.html)) を参考にしつつ、GitHubで公開されているPlaybook/Roleの構成を参考にした。

```
inventories/
    prod/
        hosts.yml       # なるべくYAMLファイルに統一し、管理しやすくする
        group_vars/
            all.yml
            site.yml
            webservers.yml
    dev/
        hosts.yml

playbooks/              # ルートにplaybookを並べるとそれ以外の設定ファイルが紛れてしまうため、playbookを分ける
    site.yml
    webservers.yml

roles/
    common/
        tasks/
            main.yml
        ...

ansible.cfg             # Pythonインタープリターのパスやログファイルの保存設定など、プロジェクトによって実行時のオプションは異なる。したがって、`ansible.cfg` をコミットするのは良い選択であると考える。
Makefile                # `ansible-playbook` の実行や、必要なroles/collectionsのインストールを行う
requirement.yml
```

### インベントリーの運用

ini形式よりもYAML形式を使う。ini形式ではインベントリファイルで host_vars を記載する際に host の後ろにスペースを開けて記述する必要がある（つまり文法が分かりづらい）など、分かりづらいため。

```ini
# inventory/hosts
[local]
localhost var_debug="debug var"
```

### ホストの動的な指定

クラウド上にサーバーを構築する場合、sshのconfigとAnsibleのインベントリの両方を指定する必要がある。DRY原則を適用するため、インベントリにはconfigファイルを読み込むPythonスクリプトを指定するのが望ましい。

## Style Rule

### 変数

Ansibleでは多用な方法で変数を指定できる。以下に挙げたものを使うことにする。

- `role defaults`: Roleのデフォルト（例: defaultsにおける、WiFiをメニューバーに表示する設定など）
- `inventory group_vars`: インベントリのグループ変数（例: 環境ごとのURL）
- `inventory host_vars`: インベントリのホスト変数（例: 環境ごとのIPアドレス）
- `vars_prompt`: プロンプトで変数を入力する場合（例: パスワード）

逆に、以下に挙げるものは基本的に使用しないことにする。

- `play vars`: 環境ごとに値を変えたい場合に不便なため。
- `play vars_files`: `play vars_prompt` よりも優先されるという仕様を忘れやすいため。
- `role vars`: `play vars_prompt` や `play vars_files` よりも優先されるという仕様を忘れやすいため。

### その他の文法

- 検索性のため、タスクはフルパスで指定する（例: `name: ansible.builtin.debug`）
- `true` と `yes` では、Ansibleの公式Collectionのドキュメントで使われている`yes` を用いる。ただし、今後方針を変えるかもしれない。

- copy or template or file or replace
  - わからん、変数を使わない場合はcopyで良さそう。
[Choosing between Ansible's copy and template modules \| Enable Sysadmin](https://www.redhat.com/sysadmin/ansibles-copy-template-modules)

- lineinfile, blockinfile or replace について、追加なら lineinefile, blockinfile を、置換なら replace を用いる。
  - lineinfileは置換だけでなく追加もするので、そのような挙動が望ましくないならreplaceを使うべき。
  - 置換の際には、[実行するたびに重複が発生する](https://logmi.jp/tech/articles/325477)ようなケースを避けるべく、正規表現では行頭から行末までを指定するのが望ましい。

## Formatting Rule

- 二重波括弧と変数名の間にスペースを挿入する（例: `{{ name }}`）
  - [API — Jinja Documentation \(3\.0\.x\)](https://jinja.palletsprojects.com/en/3.0.x/api/#basics)

## Meta Rule

- インベントリーファイルはYAMLで書く（例: `inventories/prod/hosts.yml`）

### ファイル、フォルダ

YAMLファイルの拡張子は `.yml` を用いる。

- [Tips and tricks — Ansible Documentation](https://docs.ansible.com/ansible/latest/user_guide/playbooks_best_practices.html)
- moleculeで生成されるYAMLファイルの拡張子も`.yml`である。
- copyコマンドなのでコピーするファイルは、`.yml`ではなく`.yaml`でも構わない。

Playbookのシェバンは不要。公式ドキュメントにないため。

### 運用

Makefileで運用する。

## 参考

- [Best Practices — Ansible Documentation](https://docs.ansible.com/ansible/2.8/user_guide/playbooks_best_practices.html)
- [openshift/openshift\-ansible: Install and config an OpenShift 3\.x cluster](https://github.com/openshift/openshift-ansible)
- [Azure/sonic\-mgmt: Configuration management examples for SONiC](https://github.com/Azure/sonic-mgmt)
- [containerd/contrib/ansible at main · containerd/containerd](https://github.com/containerd/containerd/tree/main/contrib/ansible)
- [Jaraxal/ansible\-aws\-example: An example Ansible playbook for creating a security group and a specified number of instances on AWS\.](https://github.com/Jaraxal/ansible-aws-example)
[openenclave/scripts/ansible at master · openenclave/openenclave](https://github.com/openenclave/openenclave/tree/master/scripts/ansible)
- [Ansible 変数の優先順位と書き方をまとめてみた \- Qiita](https://qiita.com/answer_d/items/b8a87aff8762527fb319#06-inventory-group_vars)
