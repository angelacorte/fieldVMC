package it.unibo.collektive.alchemist.device.sensors.impl

import it.unibo.alchemist.model.Environment
import it.unibo.alchemist.model.Node
import it.unibo.alchemist.model.NodeProperty
import it.unibo.alchemist.model.Position
import it.unibo.alchemist.model.molecules.SimpleMolecule
import it.unibo.collektive.alchemist.device.sensors.EnvironmentLayer
import kotlin.jvm.optionals.getOrNull

class LayerProperty<T, P : Position<P>>(
    private val environment: Environment<T, P>,
    override val node: Node<T>,
) : EnvironmentLayer, NodeProperty<T> {
    override fun cloneOnNewNode(node: Node<T>): NodeProperty<T> = LayerProperty(environment, node)

    @Suppress("UNCHECKED_CAST")
    override fun <T> getFromLayer(name: String): T =
        environment.getLayer(SimpleMolecule(name))?.getValue(environment.getPosition(node)) as T

    override fun <T> getFromLayerOrNull(name: String): T? = if (isLayerDefined(name)) getFromLayer(name) else null

    override fun isLayerDefined(name: String): Boolean = environment.getLayer(SimpleMolecule(name)) != null

    override fun <T> getFromLayerOrDefault(
        name: String,
        default: T,
    ): T = getFromLayer(name) ?: default
}
