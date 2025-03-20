package it.unibo.alchemist.actions

import it.unibo.alchemist.model.Action
import it.unibo.alchemist.model.Context
import it.unibo.alchemist.model.Environment
import it.unibo.alchemist.model.Node
import it.unibo.alchemist.model.Node.Companion.asProperty
import it.unibo.alchemist.model.Position
import it.unibo.alchemist.model.Reaction
import it.unibo.alchemist.model.actions.AbstractAction
import it.unibo.alchemist.model.molecules.SimpleMolecule
import it.unibo.collektive.alchemist.device.properties.Clock
import it.unibo.collektive.alchemist.device.properties.Cycle.BACKWARD
import it.unibo.collektive.alchemist.device.properties.Cycle.FORWARD
import it.unibo.collektive.alchemist.device.properties.impl.ExecutionClockProperty
import it.unibo.collektive.alchemist.device.sensors.impl.ResourceSensorProperty
import it.unibo.collektive.alchemist.device.sensors.impl.SuccessSensorProperty
import kotlin.math.pow

class ResourceDistribution<T, P : Position<P>>(
    private val environment: Environment<T, P>,
    private val node: Node<T>,
    private val resourceSensor: ResourceSensorProperty<T, P>,
    private val successSensor: SuccessSensorProperty<T, P>,
    private val constConsumptionRate: Double,
    private val constCompetitionRate: Double,
    private val sensorCompetitionRate: Double,
    private val vesselsAdaptationRate: Double,
) : AbstractAction<T>(node) {
    override fun cloneAction(
        node: Node<T>,
        reaction: Reaction<T>,
    ): Action<T> =
        ResourceDistribution(
            environment,
            node,
            node.asProperty<T, ResourceSensorProperty<T, P>>(),
            node.asProperty<T, SuccessSensorProperty<T, P>>(),
            constConsumptionRate,
            constCompetitionRate,
            sensorCompetitionRate,
            vesselsAdaptationRate,
        )

    override fun execute() {
        val neighbors =
            environment
                .getNeighborhood(node)
                .filterNot { n -> n.id == node.id }
                .map { it to it.asProperty<T, ExecutionClockProperty<T, P>>() }
        val current = node.asProperty<T, ExecutionClockProperty<T, P>>().currentClock()
        val parent = neighbors.firstOrNull { (n, _) -> node.getConcentration(SimpleMolecule("parent")) == n.id }
        val isRoot = node.getConcentration(SimpleMolecule("parent")) == node.id
        val children = neighbors.filter { (n, _) -> n.getConcentration(SimpleMolecule("parent")) == node.id }
        val childrenNotDone =
            children.filterNot { (_, c) ->
                c.currentClock() == Clock(time = current.time, action = BACKWARD)
            }
        if ((isRoot && childrenNotDone.isEmpty()) ||
                (current.action == BACKWARD
                    && parent != null
                    && parent.second.currentClock() == Clock(current.time, FORWARD)
                )
        ) {
            var availableResources = node.getConcentration(SimpleMolecule("resource")) as Double
            if (isRoot) { // root adds resources
                availableResources = resourceSensor.maxResource
            }
            // consume resources
            val remainingResources = availableResources - (availableResources * constConsumptionRate)
            node.setConcentration(SimpleMolecule("resource"), remainingResources as T)
            if (children.isNotEmpty()) { // should not be a leaf to evaluate weight and resource distribution
                val weightSum =
                    children.sumOf { (n, _) ->
                        n.getConcentration(SimpleMolecule("weight")) as Double
                    }
                children.forEach { (n, _) ->
                    // get the weight of the connection between the parent and the child
                    val weight = n.getConcentration(SimpleMolecule("weight")) as Double
                    // get the success of the child
                    val success = n.getConcentration(SimpleMolecule("success")) as Double
                    // update the weight of the connection
                    val newWeight = weight + vesselsAdaptationRate * (success.pow(competition()) - weight)
                    n.setConcentration(SimpleMolecule("weight"), newWeight as T)
                    // update the resource of the child based on the weight of the connection
                    val resource = remainingResources * (newWeight / weightSum)
                    n.setConcentration(SimpleMolecule("resource"), resource as T)
                }
            }
            node.asProperty<T, ExecutionClockProperty<T, P>>().nextClock()
        }
    }

    override fun getContext(): Context? = Context.NEIGHBORHOOD

    private fun competition(): Double = constCompetitionRate + sensorCompetitionRate * successSensor.getLocalSuccess()
}
