package it.unibo.alchemist.boundary.extractors

import it.unibo.alchemist.boundary.ExportFilter
import it.unibo.alchemist.boundary.exportfilters.CommonFilters
import it.unibo.alchemist.model.Actionable
import it.unibo.alchemist.model.Environment
import it.unibo.alchemist.model.Node
import it.unibo.alchemist.model.Time
import it.unibo.alchemist.model.molecules.SimpleMolecule
import it.unibo.alchemist.util.StatUtil
import org.apache.commons.math3.stat.descriptive.UnivariateStatistic

class NodeDegree
    @JvmOverloads
    constructor(
        val checkChildren: Boolean = false,
        private val filter: ExportFilter = CommonFilters.ONLYFINITE.filteringPolicy,
        aggregatorNames: List<String> = listOf("max", "mean", "stdev"),
        precision: Int = 2,
    ) : AbstractAggregatingDoubleExporter(filter, aggregatorNames, precision) {
    private companion object {
        private const val NAME = "network-degree"
    }

    override val colunmName: String
        get() = NAME

    private val aggregators: Map<String, UnivariateStatistic> =
        aggregatorNames
            .associateWith { StatUtil.makeUnivariateStatistic(it) }
            .filter { it.value.isPresent }
            .map { it.key to it.value.get() }
            .toMap()

    override val columnNames: List<String> =
        aggregators.keys
            .takeIf { it.isNotEmpty() }
            ?.map { "$NAME[$it]" }
            ?: listOf("$NAME@node-id")

    override fun <T> getData(
        environment: Environment<T, *>,
        reaction: Actionable<T>?,
        time: Time,
        step: Long
    ): Map<Node<T>, Double> =
        environment.nodes.associateWith { node ->
            val neighbors = environment.getNeighborhood(node)
            if(checkChildren) {
                neighbors.filter { n ->
                    n.getConcentration(SimpleMolecule("parent")) == node.id ||
                        node.getConcentration(SimpleMolecule("parent")) == n.id
                }.size
            } else {
                neighbors.size()
            }.toDouble()
        }
}
