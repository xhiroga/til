plugins {
    groovy
}

dependencies {
    compile("org.codehaus.groovy:groovy:2.4.10")

    testCompile("org.spockframework:spock-core:1.3-groovy-2.5") {
        exclude(module = "groovy-all")
    }
}
