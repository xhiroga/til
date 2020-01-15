plugins {
    java
    application
    groovy
}

application {
    mainClassName = "greeter.Greeter"
}

dependencies {
    compile(project(":greeting-library"))
    testCompile("org.spockframework:spock-core:1.0-groovy-2.4") {
        exclude(module = "groovy-all")
    }
}
