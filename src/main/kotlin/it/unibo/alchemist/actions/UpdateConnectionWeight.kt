package it.unibo.alchemist.actions

import it.unibo.alchemist.model.*
import it.unibo.alchemist.model.actions.AbstractAction
import it.unibo.alchemist.model.molecules.SimpleMolecule
import it.unibo.collektive.alchemist.device.sensors.SuccessSensor
import kotlin.math.pow

class UpdateConnectionWeight<T, P : Position<P>>(
    private val environment: Environment<T, P>,
    private val node: Node<T>,
    val successSensor: SuccessSensor,
    val constCompetitionRate: Double,
    val sensorCompetitionRate: Double,
    val vesselsAdaptationRate: Double,
) : AbstractAction<T>(node) {
    override fun cloneAction(
        node: Node<T>,
        reaction: Reaction<T>,
    ): Action<T> = UpdateConnectionWeight(environment, node, successSensor, constCompetitionRate, sensorCompetitionRate, vesselsAdaptationRate)

    override fun execute() {
        // todo check on the clock of the node
        environment.getNeighborhood(node).filter { n -> // children
            n.getConcentration(SimpleMolecule("parent")) == node.id
        }.forEach { n ->
            val weight = n.getConcentration(SimpleMolecule("weight")) as Double
            val success = n.getConcentration(SimpleMolecule("success")) as Double
            val newWeight = weight + vesselsAdaptationRate * (success.pow(competition()) - weight)
            n.setConcentration(SimpleMolecule("weight"), newWeight as T)
        }
    }

    override fun getContext(): Context? = Context.NEIGHBORHOOD

    private fun competition(): Double =
        constCompetitionRate + sensorCompetitionRate * successSensor.getLocalSuccess()
}
