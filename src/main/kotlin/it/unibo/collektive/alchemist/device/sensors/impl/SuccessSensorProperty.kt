@file:Suppress("UNCHECKED_CAST")

package it.unibo.collektive.alchemist.device.sensors.impl

import it.unibo.alchemist.model.Environment
import it.unibo.alchemist.model.Node
import it.unibo.alchemist.model.NodeProperty
import it.unibo.alchemist.model.Position
import it.unibo.alchemist.model.molecules.SimpleMolecule
import it.unibo.collektive.alchemist.device.sensors.SuccessSensor
import org.apache.commons.math3.random.RandomGenerator
import kotlin.jvm.optionals.getOrNull

class SuccessSensorProperty<T, P : Position<P>>(
    private val maxSuccess: Double,
    override val node: Node<T>,
    private val environment: Environment<T, P>,
    ) : SuccessSensor, NodeProperty<T> {
    override fun cloneOnNewNode(node: Node<T>): NodeProperty<T> =
        SuccessSensorProperty(maxSuccess, node, environment)

    @Suppress("UNCHECKED_CAST")
    override fun setSuccess(success: Double) =
        node.setConcentration(SimpleMolecule("success"), success as T)

    @Suppress("UNCHECKED_CAST")
    override fun setLocalSuccess(localSuccess: Double) =
        node.setConcentration(SimpleMolecule("local-success"), localSuccess as T)

    override fun getSuccess(): Double =
        node.getConcentration(SimpleMolecule("success")) as Double

    override fun getLocalSuccess(): Double =
        getFromLayer("successSource") // ?: random.nextDouble(0.0, maxSuccess)

    private fun <T> getFromLayer(name: String): T =
        environment.getLayer(SimpleMolecule(name)).getOrNull()?.getValue(environment.getPosition(node)) as T

    override fun toString(): String = this::class.simpleName!!
}
