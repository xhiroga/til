plugins {
    groovy
}

dependencies {
    compile("org.codehaus.groovy:groovy:2.4.10")

    testCompile("org.spockframework:spock-core:1.0-groovy-2.4") {
        exclude(module = "groovy-all")
    }
}
