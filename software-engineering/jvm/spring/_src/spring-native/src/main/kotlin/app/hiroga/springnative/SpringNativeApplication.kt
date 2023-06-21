package app.hiroga.springnative

import org.springframework.boot.autoconfigure.SpringBootApplication
import org.springframework.boot.runApplication

@SpringBootApplication
class SpringNativeApplication

fun main(args: Array<String>) {
	runApplication<SpringNativeApplication>(*args)
}
