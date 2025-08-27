package it.unibo.collektive.coordination

import it.unibo.alchemist.collektive.device.DistanceSensor
import it.unibo.collektive.aggregate.api.Aggregate
import it.unibo.collektive.aggregate.api.operators.share
import it.unibo.collektive.field.Field
import it.unibo.collektive.field.Field.Companion.fold
import java.io.Serializable

/**
 * Elect the leader in an area limited by the [radius], based on the [localStrength] of the node.
 */
fun <ID : Any, C : Comparable<C>> Aggregate<ID>.boundedElection(
    distanceSensor: DistanceSensor,
    localStrength: C,
    radius: Double,
): ID {

    val local: Candidacy<ID, C> = Candidacy(localStrength, 0.0, localId)
    return share(local) { candidates ->
        val candidate =
            with(distanceSensor) {
                candidates.alignedMap(distances()) { c, m -> Candidacy(c.strength, c.distance + m, c.leaderId) }
            }
        val field: Field<ID, Candidacy<ID, C>?> =
            candidate
                .mapWithId { id, c -> c.takeUnless { id == localId || it.distance > radius } }
        field.fold(local) { accumulator, newValue ->
            when {
                newValue == null -> accumulator
                else -> minOf(accumulator, newValue)
            }
        }
    }.leaderId
}

data class Candidacy<ID : Any, C: Comparable<C>>(
    val strength: C,
    val distance: Double,
    val leaderId: ID,
) : Serializable, Comparable<Candidacy<ID, C>> {
    override fun compareTo(other: Candidacy<ID, C>): Int =
        Comparator<Candidacy<ID, C>> { a, b -> b.strength.compareTo(a.strength) }
            .thenBy { it.distance }
            .thenBy {
                when (it.leaderId) {
                    is Comparable<*> -> it.leaderId
                    else -> 0
                }
            }.compare(this, other)
}
