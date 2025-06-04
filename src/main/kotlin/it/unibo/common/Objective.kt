package it.unibo.common

import it.unibo.alchemist.model.Environment
import it.unibo.alchemist.util.Environments.networkDiameterByHopDistance
import it.unibo.common.DataRetriever.retrieveData
import it.unibo.common.NetworkMetrics.networkDensity
import it.unibo.common.NetworkMetrics.networkHub
import it.unibo.common.NetworkMetrics.nodesDegree
import org.apache.commons.math3.stat.descriptive.moment.GeometricMean
import kotlin.math.absoluteValue

/**
 * The goal for the optimization.
 */
class Goal : (Environment<*, *>) -> Double {
    val metrics: List<String> =
        listOf(
            "nodes",
            "network-hub-xCoord",
            "network-hub-yCoord",
            "network-density",
            "network-diameter",
            "nodes-degree[mean]",
        )

    val target = target("self-construction-classic-vmc", metrics)

    override fun invoke(env: Environment<*, *>): Double = env.minimize(target)
}

/**
 * The target value for the optimization.
 * It is the geometric mean of the mean values of the metrics for the classic-vmc experiment.
 * The metrics are:
 * - number of nodes;
 * - x and y coordinates of the network hub;
 * - network diameter;
 * - network degree.
 */
fun target(
    experimentName: String,
    metrics: List<String> = Goal().metrics,
): Map<String, Double> =
    retrieveData(
        listOf(experimentName),
        "data",
    ).mapKeys { (key, _) -> key.removePrefix("$experimentName@") }
        .filterKeys { it in metrics }

/**
 * The function to minimize.
 * It is the difference between the target value and the current value.
 * The current value is the geometric mean of the metrics for the given [Environment].
 * The difference is taken modulo 2 to avoid negative values.
 * The metric values are:
 * - number of nodes;
 * - x and y coordinates of the network hub;
 * - network diameter;
 * - network degree.
 */
fun Environment<*, *>.minimize(target: Map<String, Double>): Double =
    networkHub().let { (xCoord, yCoord) ->
        val current =
            mapOf(
                "nodes" to nodeCount.toDouble(),
                "network-hub-xCoord" to xCoord,
                "network-hub-yCoord" to yCoord,
                "network-density" to networkDensity(),
                "network-diameter" to networkDiameterByHopDistance(),
                "nodes-degree[mean]" to nodesDegree().average(),
            )
        val absoluteDifference =
            target.map { (metric, target) ->
                val relative = current[metric] ?: error("Metric $metric not found")
                // Add 1 to avoid values as 0
                ((target - relative).absoluteValue + 1).also {
                    require(it >= 1) { "The difference between target and current value should not be less than 1, but is $it" }
                }
            }
        GeometricMean().evaluate(absoluteDifference.toDoubleArray())
    }
