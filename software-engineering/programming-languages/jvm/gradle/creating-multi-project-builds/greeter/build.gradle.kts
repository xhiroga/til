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
    testCompile("org.spockframework:spock-core:2.3-groovy-4.0") {
        exclude(module = "groovy-all")
    }
}
