package it.unibo.common

import it.unibo.alchemist.model.Environment
import kotlin.collections.component1
import kotlin.collections.component2
import kotlin.math.max
import kotlin.math.min

object NetworkMetrics {
    /**
     * The network density.
     * The network density is the average number of nodes per unit area.
     * The area is the rectangle given by the outermost nodes in the network.
     * The result is the average of the network densities of all the nodes in the network.
     */
    fun <T> Environment<T, *>.networkDensity(): Double {
        data class BoundingBox(
            val minX: Double = Double.Companion.POSITIVE_INFINITY,
            val maxX: Double = Double.Companion.NEGATIVE_INFINITY,
            val minY: Double = Double.Companion.POSITIVE_INFINITY,
            val maxY: Double = Double.Companion.NEGATIVE_INFINITY,
        )
        val boundingBox = this.fold(BoundingBox()) { bb, node ->
            val (x, y) = getPosition(node).coordinates
            BoundingBox(
                min(x, bb.minX),
                max(x, bb.maxX),
                min(y, bb.minY),
                max(y, bb.maxY),
            )
        }
        val area = (boundingBox.maxX - boundingBox.minX) * (boundingBox.maxY - boundingBox.minY)
        if (area <= 0 || area.isInfinite()) return Double.Companion.NaN
        return nodeCount / area
    }

    /**
     * The degree of the nodes in the network.
     * The degree of a node is the number of neighbors it has.
     */
    fun <T> Environment<T, *>.nodesDegree(): List<Double> =
        nodes
            .map { n ->
                val neighbors = getNeighborhood(n)
                neighbors.size().toDouble()
            }

    /**
     * The network hub coordinates.
     * The x and y coordinates are the average of the x and y coordinates of the nodes in the network.
     */
    fun <T> Environment<T, *>.networkHub(): Pair<Double, Double> =
        fold(0.0 to 0.0) { acc, next ->
            val nodePos = this.getPosition(next).coordinates.map { it + 10 } // Add 10 to avoid negative positions
            acc.first + nodePos[0] to acc.second + nodePos[1]
        }.let { sum ->
            sum.first / this.nodeCount to sum.second / this.nodeCount
        }
}