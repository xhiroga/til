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
    testCompile("org.spockframework:spock-core:1.3-groovy-2.5") {
        exclude(module = "groovy-all")
    }
}
