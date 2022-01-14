# Awesome macOS Provisioning

A self curated list of awesome macOS provisioning scripts.


---

## SetUp Tools

- [mathiasbynens/dotfiles](https://github.com/mathiasbynens/dotfiles) - GitHubで最もスターの多いdotfiles。[dotfiles上位勢はmacOS向けが占めているらしい](https://zenn.dev/yutakatay/articles/try-dotfiles-01)。defaultsコマンドの設定でもお世話になりました。

- [thoughtbot/laptop: A shell script to set up a macOS laptop for web and mobile development\.](https://github.com/thoughtbot/laptop/) - Web・Mobile開発のためのSetUpスクリプト。最終更新は2021年初頭。アプリケーションのインストールのみ（defaultsコマンド未使用）で、かなりシンプルだがスター8Kある。

- [minamarkham/formation: 💻 macOS setup script for front\-end development](https://github.com/minamarkham/formation) - Web開発のためのSetUpスクリプト。ZshでなくBashだったり、Dockerの代わりにVagrantだったり、Gulp使ってたり多少古い感じもするが、知らなかったツールがたくさんある。最終更新は2019年。Mathiasのdotfilesと上述のlaptopを参考にしている。

- [geerlingguy/mac\-dev\-playbook: Mac setup and configuration via Ansible\.](https://github.com/geerlingguy/mac-dev-playbook) - macOSセットアップでよくある操作（Dockからアプリを除外とか）をAnsibleのRoleにまとめているっぽい？Ansible読めないとありがたみがわからないが複数OSを設定するなら個人的にはこの方向性だと思う。

- [MikeMcQuaid/strap: 👢 Bootstrap your macOS development system\.](https://github.com/MikeMcQuaid/strap) - Featureを列挙してくれているのが親切。ユーザーごとのdotfiles, Brewfileを勝手に設定してくれる発想が面白い。構成が個人ごとに異なりがちなdotfilesだが、このようなSetUpスクリプトに呼ばれることを前提として標準化されていくのではないか。

- [marlonrichert/\.config: ⚙️ Simple & efficient dotfiles for macOS & Ubuntu](https://github.com/marlonrichert/.config) - macOSとUbuntuに対応しているSetUpスクリプト兼dotfiles。VSCodeやKarabinerなど利用ツールが私と似ており親近感がある。installがmake経由なのも、複数OS対応を考えるとやはり...という感想。

- [isank/workstation: My work setup, tools, shell scripts and etc\.](https://github.com/isank/workstation) - なんとBrewfileでさえない、SetUp手順書。だけど知らないツールがそこそこある。

- [ptb/mac\-setup](https://github.com/ptb/mac-setup) - Mathiasのdotfilesでも紹介されているdotfiles。最終更新は2018年。


## etc

- [rgcr/m\-cli:  Swiss Army Knife for macOS](https://github.com/rgcr/m-cli) - SetUpスクリプトではありません。iTunesなどのmacOSアプリケーションをCLIからいい感じに操作するためのユーティリティ。

---

## References

- [Top 10 Shell Mac Projects \(Sep 2021\)](https://www.libhunt.com/l/shell/t/mac)
