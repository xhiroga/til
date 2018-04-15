# Maven

## プロジェクトの構築
```console
mvn archetype:generate
# 引数でMainクラスを指定する場合、pom.xmlでの指定は不要
mvn compile
mvn exec:java -Dexec.mainClass=cc.hiroga.App
```
* groupId: プロジェクトのルートパッケージ名  
ex) cc.hiroga  
* artifactId: プロジェクトのパッケージ名  
ex) sampleApp

## 参考
[Eclipseは使わない！Mavenでサンドボックス作るよー！](http://hiroga.hatenablog.com/entry/2017/12/04/234930)