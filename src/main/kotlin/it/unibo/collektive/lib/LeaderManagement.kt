package it.unibo.collektive.lib

import it.unibo.alchemist.collektive.device.DistanceSensor
import it.unibo.collektive.aggregate.api.Aggregate
import it.unibo.collektive.alchemist.device.sensors.LeaderSensor
import it.unibo.collektive.alchemist.device.sensors.ResourceSensor
import it.unibo.collektive.coordination.boundedElection
import it.unibo.collektive.stdlib.spreading.distanceTo

/**
 * Elect the leader of the current node.
 */
fun <ID : Comparable<ID>> Aggregate<ID>.chooseLeader(
    distanceSensor: DistanceSensor,
    leaderSensor: LeaderSensor,
    resourceSensor: ResourceSensor,
): ID = boundedElection(distanceSensor, resourceSensor.getResource(), leaderSensor.leaderRadius)

/**
 * Find the potential of the current node.
 */
fun <ID : Comparable<ID>> Aggregate<ID>.findPotential(
    distanceSensor: DistanceSensor,
    leader: Boolean,
): Double = distanceTo(leader) { with(distanceSensor) { distances() } }

/**
 * Check if the current node is the leader.
 */
fun <ID : Comparable<ID>> Aggregate<ID>.isLeader(
    distanceSensor: DistanceSensor,
    leaderSensor: LeaderSensor,
    resourceSensor: ResourceSensor,
): Boolean = (chooseLeader(distanceSensor, leaderSensor, resourceSensor) == localId).also { leaderSensor.setLeader(it) }
