# Maven

## プロジェクトの構築
```Console
mvn archetype:generate
# 引数でMainクラスを指定する場合、pom.xmlでの指定は不要
mvn compile
mvn exec:java -Dexec.mainClass=cc.hiroga.App
```
* groupId: プロジェクトのルートパッケージ名  
ex) cc.hiroga  
* artifactId: プロジェクトのパッケージ名  
ex) sampleApp

## Webアプリケーションプロジェクトの構築
```Console
mvn archetype:generate -DgroupId=cc.hiroga.testbbs -DartifactId=testbbs -DarchetypeArtifactId=maven-archetype-webapp
# archetypeArtifactIdパラメータを明示的に指定する(これまではmaven-archetype-quickstart が使用されていた)
# src/main/java を手動で作成する
```

## 参考
[Eclipseは使わない！Mavenでサンドボックス作るよー！](http://hiroga.hatenablog.com/entry/2017/12/04/234930)