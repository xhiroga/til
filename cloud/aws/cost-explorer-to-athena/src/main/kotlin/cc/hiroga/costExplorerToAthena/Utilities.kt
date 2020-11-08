@file:Suppress("FunctionName")
package cc.hiroga.costExplorerToAthena

import software.amazon.awscdk.core.App
import software.amazon.awscdk.core.Construct
import software.amazon.awscdk.core.Stack

fun synthesize(configureApp: App.() -> Unit): Unit =
    App().run {
        configureApp().also { synth() }
    }

fun Construct.Stack(name: String, configureStack: Stack.() -> Unit): Stack =
    Stack(this, name).also(configureStack)