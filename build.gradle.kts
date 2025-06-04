import org.jetbrains.kotlin.gradle.tasks.KotlinCompile
import java.awt.GraphicsEnvironment
import java.io.ByteArrayOutputStream
import java.util.*

plugins {
    application
    alias(libs.plugins.gitSemVer)
    alias(libs.plugins.collektive)
    alias(libs.plugins.kotlin.jvm)
    alias(libs.plugins.kotlin.qa)
    alias(libs.plugins.multiJvmTesting)
    alias(libs.plugins.taskTree)
}

repositories {
    mavenCentral()
}

sourceSets {
    main {
        dependencies {
            implementation(libs.bundles.alchemist)
            implementation(libs.bundles.collektive)
            implementation(kotlin("reflect"))
            implementation(libs.caffeine)
            implementation(libs.rrmxmx)
            implementation(libs.symmetric.matrix)
        }
    }
}

multiJvm {
    jvmVersionForCompilation.set(17)
}

dependencies {
    implementation(kotlin("stdlib-jdk8"))
    if (!GraphicsEnvironment.isHeadless()) {
        implementation("it.unibo.alchemist:alchemist-swingui:${libs.versions.alchemist.get()}")
    }
}

// Heap size estimation for batches
val maxHeap: Long? by project
val heap: Long =
    maxHeap ?: if (System.getProperty("os.name").lowercase().contains("linux")) {
        ByteArrayOutputStream()
            .use { output ->
                exec {
                    executable = "bash"
                    args = listOf("-c", "cat /proc/meminfo | grep MemAvailable | grep -o '[0-9]*'")
                    standardOutput = output
                }
                output.toString().trim().toLong() / 1024
            }.also { println("Detected ${it}MB RAM available.") } * 9 / 10
    } else {
        // Guess 16GB RAM of which 2 used by the OS
        14 * 1024L
    }
val taskSizeFromProject: Int? by project
val taskSize = taskSizeFromProject ?: 5120000
val alchemistGroupBatch = "Run batch simulations"
val alchemistGroupGraphic = "Run graphic simulations with Alchemist"
val alchemistGroupOptimizer = "Run optimizer simulations with Alchemist"

/*
 * This task is used to run all experiments in sequence
 */
val runAllGraphic by tasks.register<DefaultTask>("runAllGraphic") {
    group = alchemistGroupGraphic
    description = "Launches all simulations with the graphic subsystem enabled"
}
val runAllBatch by tasks.register<DefaultTask>("runAllBatch") {
    group = alchemistGroupBatch
    description = "Launches all experiments in batch mode"
}
val runAllOptimizer by tasks.register<DefaultTask>("runAllOptimizer") {
    group = alchemistGroupOptimizer
    description = "Launches all experiments with the optimizer enabled"
}

fun String.capitalizeString(): String =
    this.replaceFirstChar {
        if (it.isLowerCase()) {
            it.titlecase(
                Locale.getDefault(),
            )
        } else {
            it.toString()
        }
    }

/*
 * Scan the folder with the simulation files, and create a task for each one of them.
 */
File(rootProject.rootDir.path + "/src/main/yaml")
    .listFiles()
    ?.filter { it.extension == "yml" }
    ?.sortedBy { it.nameWithoutExtension }
    ?.forEach {
        fun basetask(
            name: String,
            additionalConfiguration: JavaExec.() -> Unit = {},
        ) = tasks.register<JavaExec>(name) {
            description = "Launches graphic simulation ${it.nameWithoutExtension}"
            mainClass.set("it.unibo.alchemist.Alchemist")
            classpath = sourceSets["main"].runtimeClasspath
            args("run", it.absolutePath)
            if (System.getenv("CI") == "true") {
                args(
                    "--override",
                    "terminate: { type: AfterTime, parameters: [2] } ",
                )
            } else {
                this.additionalConfiguration()
            }
        }
        val capitalizedName = it.nameWithoutExtension.capitalizeString()
        val graphic by basetask("run${capitalizedName}Graphic") {
            group = alchemistGroupGraphic
            args(
                "--override",
                "monitors: { type: SwingGUI, parameters: { graphics: effects/${it.nameWithoutExtension}.json } }",
                "--override",
                "launcher: { parameters: { batch: [], autoStart: false } }",
                "--override",
                """
                variables:
                  seed: &seed
                    min: 0
                    max: 10
                    step: 1
                    default: 0
                """.trimIndent(),
                "--verbosity",
                "error",
            )
        }
        runAllGraphic.dependsOn(graphic)
        val batch by basetask("run${capitalizedName}Batch") {
            group = alchemistGroupBatch
            description = "Launches batch experiments for $capitalizedName"
            maxHeapSize = "${minOf(heap.toInt(), Runtime.getRuntime().availableProcessors() * taskSize)}m"
            File("data").mkdirs()
            args(
                "--verbosity",
                "error",
            )
            if (capitalizedName.startsWith("SelfHealing")) {
                args(
                    "--override",
                    """
                    variables:
                      metrics: &metrics
                        formula: |
                          it.unibo.common.TerminationMetrics()
                        language: kotlin
                    
                    launcher:
                      type: DefaultLauncher
                      parameters: {
                        batch: ["seed", "origin"],
                        autoStart: true,
                      }
                      
                    terminate: { type: AfterTime, parameters: [1000] }
                    """.trimIndent(),
                )
            } else if (capitalizedName.endsWith("VMC") && !capitalizedName.startsWith("SelfHealing")) {
                args(
                    "--override",
                    """
                    variables:
                    metrics: &metrics
                        formula: |
                        it.unibo.common.TerminationMetrics()
                        language: kotlin
                        
                    terminate:
                    type: MetricsStableForTime
                    parameters: {
                        stableForTime: 30.0,
                        timeIntervalToCheck: 2.0,
                        equalTimes: 3,
                        metricsToCheck: *metrics,
                    }
                    
                    launcher:
                    type: DefaultLauncher
                    parameters: {
                        batch: ["seed"],
                        autoStart: true,
                    }
                    """.trimIndent(),
                )
            } else {
                args(
                    "--override",
                    """
                    launcher:
                      type: DefaultLauncher
                      parameters: {
                        batch: ["seed", "initialNodes"],
                        autoStart: true,
                      }
                    """.trimIndent(),
                )
            }
        }
        runAllBatch.dependsOn(batch)
        if (capitalizedName.endsWith("Optimizer")) {
            val optimizer by basetask("run${capitalizedName}") {
                setDependsOn(listOf("runSelfConstructionClassicVMCBatch"))
                group = alchemistGroupOptimizer
                description = "Launches Nelder Mead parameters optimizer for $capitalizedName"
                maxHeapSize = "${minOf(heap.toInt(), Runtime.getRuntime().availableProcessors() * taskSize)}m"
                File("data").mkdirs()
            }
            runAllOptimizer.dependsOn(optimizer)
        }
    }

tasks.withType(KotlinCompile::class).all {
    compilerOptions {
        allWarningsAsErrors = false
    }
}
