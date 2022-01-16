package cc.hiroga

import com.fasterxml.jackson.databind.SerializationFeature
import io.ktor.application.Application
import io.ktor.application.call
import io.ktor.application.install
import io.ktor.client.HttpClient
import io.ktor.client.engine.cio.CIO
import io.ktor.features.CallLogging
import io.ktor.features.ContentNegotiation
import io.ktor.http.ContentType
import io.ktor.jackson.jackson
import io.ktor.response.respond
import io.ktor.response.respondText
import io.ktor.routing.get
import io.ktor.routing.post
import io.ktor.routing.routing
import io.ktor.server.engine.commandLineEnvironment
import io.ktor.server.engine.embeddedServer
import io.ktor.server.netty.Netty
import io.ktor.util.KtorExperimentalAPI
import org.koin.ktor.ext.inject
import org.koin.ktor.ext.installKoin
import java.util.*


// fun main(args: Array<String>): Unit = io.ktor.server.netty.EngineMain.main(args)
fun main(args: Array<String>) {
    embeddedServer(Netty, commandLineEnvironment(args)).start()
}

@KtorExperimentalAPI
@Suppress("unused") // Referenced in application.conf
@kotlin.jvm.JvmOverloads
fun Application.module(testing: Boolean = false) {
    install(CallLogging)

    install(ContentNegotiation) {
        jackson {
            // enable indenting json
            configure(SerializationFeature.INDENT_OUTPUT, true)
        }
    }

    // installKoin(listOf(helloAppModule), logger = SLF4JLogger())
    installKoin(listOf(mainModule))

    val random by inject<Random>()

    // https://ktor.io/clients/http-client/engines.html
    val client = HttpClient(CIO) {
        engine {
            maxConnectionsCount = 1000 // Maximum number of socket connections.
            endpoint.apply {
                maxConnectionsPerRoute = 100 // Maximum number of requests for a specific endpoint route.
                pipelineMaxSize = 20 // Max number of opened endpoints.
                keepAliveTime = 5000 // Max number of milliseconds to keep each connection alive.
                connectTimeout = 5000 // Number of milliseconds to wait trying to connect to the server.
                connectRetryAttempts = 5 // Maximum number of attempts for retrying a connection.
            }
        }
    }

    routing {
        get("/") {
            call.respondText("HELLO WORLD!", contentType = ContentType.Text.Plain)
        }

        post("/chomado") {
            val TEST_CHOMADO_PIC = "https://pbs.twimg.com/profile_images/1086536800791740417/uvkmvoBk_400x400.jpg"
            val attachment = SlackResponseAttachement(TEST_CHOMADO_PIC)
            val response = SlackResponse(
                "in_channel",
                "(*ﾟ▽ﾟ* っ)З. (*ﾟ▽ﾟ* っ)З ‬",
                arrayOf(attachment)
            )

            call.respond(response )
        }

    }
}

