# 開発環境について

# Version管理
## pyenv
複数のバージョンのPythonを使い分けられる。同じバージョンで違う環境を使いたい場合、pyenv-vertualenvを導入する。
```
pyenv versions
pyenv install -l # インストールできる環境の確認
pyenv install 3.7.0
```

## viertualenv
pyenv-virtualenvとは別の、pyenv以前からあった仮想環境構築ツール。

## Anaconda
```
# 仮想環境の作成
conad create -n opencv python=3.5 anaconda
# 仮想環境のactivate/deactivate
source activate opencv
source deactivate opencv
```
※最新バージョンのAnacondaを使うと、pyenvと一緒に使ったときにactivateが失敗することがある。

# Package管理
## directory
site-packagesディレクトリにパッケージを置く。  

SystemのPythonの場合
```
>>> import site
>>> site.getsitepackages()['/usr/local/Cellar/python@2/2.7.15_1/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages', '/usr/local/Cellar/python@2/2.7.15_1/Frameworks/Python.framework/Versions/2.7/lib/site-python']
```

pyenvの場合: Versionごとに異なる。  
```
['/Users/hiroaki/.pyenv/versions/3.7.0/lib/python3.7/site-packages']
```

## requirements.txt
`pip freeze > requirements.txt`