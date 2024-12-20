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
import it.unibo.alchemist.model.times.DoubleTime
import it.unibo.collektive.alchemist.device.properties.Clock
import it.unibo.collektive.alchemist.device.properties.Cycle.FORWARD
import it.unibo.collektive.alchemist.device.properties.Cycle.SPAWNING
import it.unibo.collektive.alchemist.device.properties.impl.ExecutionClockProperty
import it.unibo.collektive.alchemist.device.sensors.impl.LocationSensorProperty
import it.unibo.collektive.alchemist.device.sensors.impl.RandomNodeProperty
import it.unibo.common.calculateAngle
import it.unibo.common.minus
import it.unibo.common.plus
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.nextUp
import kotlin.math.sin

class Spawn<T : Any, P : Position<P>>(
    private val environment: Environment<T, P>,
    private val node: Node<T>,
    private val randomGenerator: RandomNodeProperty<T>,
    private val locationSensor: LocationSensorProperty<T, P>,
    private val cloningRange: Double,
    private val resourceThreshold: Double,
    private val maxChildren: Int,
) : AbstractAction<T>(node) {
    override fun cloneAction(
        node: Node<T>,
        reaction: Reaction<T>,
    ): Action<T> =
        Spawn(
            environment,
            node,
            randomGenerator,
            node.asProperty<T, LocationSensorProperty<T, P>>(),
            cloningRange,
            resourceThreshold,
            maxChildren,
        )

    override fun execute() {
        val allNodes = environment.nodes.map { it to it.asProperty<T, ExecutionClockProperty<T, P>>() }
        val maxLeaf = allNodes.filter { (n, _) -> n.getConcentration(SimpleMolecule("leaf")) == true }
            .maxBy { (n, _) -> n.getConcentration(SimpleMolecule("resource")) as Double }
        val localResource = node.getConcentration(SimpleMolecule("resource")) as Double
        val current = node.asProperty<T, ExecutionClockProperty<T, P>>().currentClock()
        val nodesNotInForward = allNodes.filterNot { (_, s) ->
            s.currentClock() == Clock(time = current.time, action = FORWARD)
        }
        val isLeaf = allNodes
            .filter { (n, _) -> n.id == node.id }
            .none { (n, _) -> n.getConcentration(SimpleMolecule("parent")) == node.id }
        val SPAWNING = allNodes.filter { (_, c) -> c.currentClock().action == SPAWNING }
        if(nodesNotInForward.isEmpty() && maxLeaf.first.id == node.id && localResource >= (resourceThreshold * 2)) { // && current.time % 21 == 0
            val spawnableChildren = (localResource / resourceThreshold).toInt().coerceIn(1, maxChildren)
            val localPosition: Pair<Double, Double> = locationSensor.coordinates()
            val neighborPosition: List<Pair<Double, Double>> = locationSensor.surroundings()
            val relativePositions: List<Pair<Double, Double>> = neighborPosition.map { it - localPosition}
            val angles = relativePositions.map { atan2(it.second, it.first) }.sorted()
            repeat(spawnableChildren) {
                val angle = calculateAngle(angles, randomGenerator, maxChildren)
                val x = cloningRange * cos(angle)
                val y = cloningRange * sin(angle)
                val absoluteDestination = if( it % 2 == 0 ) localPosition + (x to y) else localPosition + (-x to y)
                spawn(absoluteDestination)
            }
        } else if (isLeaf && SPAWNING.isEmpty()) {
            node.asProperty<T, ExecutionClockProperty<T, P>>().cannotSpawn()
        }
    }

    override fun getContext(): Context? = Context.NEIGHBORHOOD

    private fun spawn(coordinate: Pair<Double, Double>) {
        val spawningTime = environment.simulation.time + DoubleTime(randomGenerator.nextRandomDouble(0.0.nextUp() .. 0.1))
        val cloneOfThis = node.cloneNode(spawningTime)
        val nodeClock = node.asProperty<T, ExecutionClockProperty<T, P>>().currentClock()
        cloneOfThis.setConcentration(SimpleMolecule("parent"), node.id as T)
        cloneOfThis.setConcentration(SimpleMolecule("weight"), 0.1 as T)
        cloneOfThis.setConcentration(SimpleMolecule("resource"), 0.0 as T)
        cloneOfThis.asProperty<T, ExecutionClockProperty<T, P>>().justSpawned(nodeClock.time + 1)
        val updatedPosition = environment.makePosition(*coordinate.toList().toTypedArray())
        environment.addNode(cloneOfThis, updatedPosition)
    }
}
