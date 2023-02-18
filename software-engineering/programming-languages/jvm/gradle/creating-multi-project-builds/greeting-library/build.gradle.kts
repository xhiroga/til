plugins {
    groovy
}

dependencies {
    compile("org.codehaus.groovy:groovy:2.4.10")

    testCompile("org.spockframework:spock-core:2.3-groovy-4.0") {
        exclude(module = "groovy-all")
    }
}
