package it.unibo.alchemist.boundary.extractors

import it.unibo.alchemist.boundary.ExportFilter
import it.unibo.alchemist.model.Actionable
import it.unibo.alchemist.model.Environment
import it.unibo.alchemist.model.Node
import it.unibo.alchemist.model.Time
import kotlin.Double.Companion.NaN

class NetworkDensity
    @JvmOverloads
    constructor(
        filter: ExportFilter,
        aggregators: List<String>,
        precision: Int = 2,
    ) : AbstractAggregatingDoubleExporter(filter, aggregators, precision) {
        private companion object {
            private const val NAME: String = "network-density"
        }

        override val columnName: String = NAME

        override fun <T> getData(
            environment: Environment<T, *>,
            reaction: Actionable<T>?,
            time: Time,
            step: Long,
        ): Map<Node<T>, Double> {
            var outers: MutableMap<String, Double> =
                listOf<String>("top", "bottom", "right", "left")
                    .associateWith { NaN }
                    .toMutableMap()
            return environment.nodes.associateWith { n ->
                val nodePos = environment.getPosition(n).coordinates.map { it + 10 } // Add 10 to avoid negative positions
                outers =
                    outers.mapValues { (key, value) ->
                        when {
                            key == "right" && (value.isNaN() || nodePos[0] > value) -> nodePos[0]
                            key == "left" && (value.isNaN() || nodePos[0] < value) -> nodePos[0]
                            key == "top" && (value.isNaN() || nodePos[1] > value) -> nodePos[1]
                            key == "bottom" && (value.isNaN() || nodePos[1] < value) -> nodePos[1]
                            else -> value
                        }
                    } as MutableMap<String, Double>
                // Calculate the area of the rectangle given by the outermost nodes
                val area = (outers["right"]!! - outers["left"]!!) * (outers["top"]!! - outers["bottom"]!!)
                environment.nodeCount / area
            }
        }
    }
