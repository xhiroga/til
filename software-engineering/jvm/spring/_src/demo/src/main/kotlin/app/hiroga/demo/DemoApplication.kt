package app.hiroga.demo

import org.springframework.boot.autoconfigure.SpringBootApplication
import org.springframework.boot.runApplication
import org.springframework.context.ApplicationContext
import org.springframework.context.annotation.Bean
import org.springframework.boot.CommandLineRunner

@SpringBootApplication
class DemoApplication {

	@Bean // @Beanは起動時に自動で実行される。
	fun commandLineRunner(ctx: ApplicationContext): CommandLineRunner {
		return CommandLineRunner {
			println("Let's inspect the beans provided by Spring Boot:")

			val beanNames = ctx.beanDefinitionNames
			beanNames.sort()
			for (beanName in beanNames) {
				println(beanName)
			}
		}
	}
}

fun main(args: Array<String>) {
	runApplication<DemoApplication>(*args)
}
