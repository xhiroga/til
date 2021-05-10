# programming-haskell

## setup IntelliJ

- see [How do I set up IntelliJ to build Haskell projects with Stack?](https://stackoverflow.com/questions/37754956/how-do-i-set-up-intellij-to-build-haskell-projects-with-stack)
- install `stack install hspec-discover`

### note

- JetBrains 公式を含め、他の Haskell Plugin がインストールされているとエラーが起きる。

## test

```
# hspec-discover を利用し、テスト対象のファイルを自動で見つけている。 
stack test
```


## references

- [プログラミング Haskell 第 2 版](https://amzn.to/36si782)
- [テストフレームワーク (hspec)](https://haskell.e-bigmoon.com/stack/test/hspec.html)