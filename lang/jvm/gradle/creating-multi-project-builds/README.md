# Gradle Multi-project Builds

## Note

* 手動で build.gradle.kts を作成した場合に Gradleプロジェクトとして認知させるには
    * Intellijを再起動
    * Event Log から Import Gradle Project を選択すればよい
* Intellij のメニューで New Module → Gradle(Kotlin DSL)は、SubProject作成でエラーを起こす(2020-01-13)
* Intellij のProject Filesの表示がおかしい（mainとtestが別々の親Module Groupに属しているように見える、等）
    * .ideaを削除してからProjectを再度importする。


## Reference

* https://guides.gradle.org/creating-multi-project-builds
* https://www.jetbrains.com/help/idea/gradle.html
