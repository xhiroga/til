# gradle

1. Create project. Some options.
   1. Run `gradle init --type java-application`
   2. Run `Java: Create Java Project` in VSCode command palette.
   3. Create new project by Intellij.
2. Run project.
   1. Run `./gradlew run`. Note `run` command is provided by plugin Application.
   2. Run by VSCode debug menu.
   3. Run by Intellij.

## Note

* To Run/Debug main class from vscode, .classpath is needed.
* Language Support for Java usually creates .classpath, but sometime not.
  * If not, remove workspaceStorage and reboot VSCode.
  * Or run `./gradlew eclipse` and generate .classpath.
* tasks.json makes gradle task executable from the command palette.

## Reference

https://spring.io/guides/gs/gradle/
https://github.com/redhat-developer/vscode-java/issues/132
