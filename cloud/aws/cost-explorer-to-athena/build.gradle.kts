plugins {
    kotlin("jvm") version "1.3.50"
    application
}

tasks {
    named<Wrapper>("wrapper") {
        gradleVersion = "5.6.2"
    }

    compileKotlin {
        kotlinOptions.jvmTarget = JavaVersion.VERSION_1_8.toString()
    }

    "run" {
        // Allow skipping code execution conditionally
        outputs.dir("cdk.out")
    }
}

application {
    mainClassName = "jp.justincase.cdkdsl.example.Main"
}

repositories {
    jcenter()
    maven(url = "https://dl.bintray.com/justincase/aws-cdk-kotlin-dsl")
}

dependencies {
    implementation(kotlin("stdlib"))
    implementation("jp.justincase.aws-cdk-kotlin-dsl", "s3", "1.11.0.DEVPREVIEW-0.5.2")

}