package cc.hiroga

import org.koin.dsl.module.module
import java.security.SecureRandom
import java.util.*

val mainModule = module(createOnStart = true) {
    single<Random> { SecureRandom() }

    // single { HelloMessageData() }
    // single<HelloService> { HelloServiceImpl(get()) }

    // singleBy<HelloService, HelloServiceImpl>()
    // single<HelloRepository>()
}
