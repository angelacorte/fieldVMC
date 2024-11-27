package it.unibo.collektive.vmc

import it.unibo.alchemist.collektive.device.DistanceSensor
import it.unibo.collektive.aggregate.api.Aggregate
import it.unibo.collektive.aggregate.api.operators.share
import it.unibo.collektive.alchemist.device.sensors.DeviceSpawn
import it.unibo.collektive.alchemist.device.sensors.EnvironmentVariables
import it.unibo.collektive.alchemist.device.sensors.LeaderSensor
import it.unibo.collektive.alchemist.device.sensors.LocationSensor
import it.unibo.collektive.alchemist.device.sensors.RandomGenerator
import it.unibo.collektive.alchemist.device.sensors.ResourceSensor
import it.unibo.collektive.alchemist.device.sensors.SuccessSensor
import it.unibo.collektive.coordination.findParent
import it.unibo.collektive.field.Field.Companion.fold
import it.unibo.collektive.lib.convergeSuccess
import it.unibo.collektive.lib.findPotential
import it.unibo.collektive.lib.isLeader
import it.unibo.collektive.lib.obtainLocalSuccess
import it.unibo.collektive.lib.spreadResource
import kotlin.math.PI
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin

/**
 * Entrypoint of the VMC algorithm, using spawning and destroying after stability policies.
 */
context(
    EnvironmentVariables,
    DeviceSpawn,
    LocationSensor,
    LeaderSensor,
    ResourceSensor,
    SuccessSensor,
    RandomGenerator,
    DistanceSensor
)
fun Aggregate<Int>.withSpawning(): Double = spawnAndDestroyAfterStability()

/**
 * Spawns a new node or destroys an old one if the conditions are met.
 * The node is spawned if the local resources are above the lower bound threshold,
 * if it has less than a maximum threshold of children and the neighborhood is stable.
 * The node is destroyed if the local resources are below the lower bound,
 * if it is not father of any node and the neighborhood is stable.
 */
context(
    EnvironmentVariables,
    DeviceSpawn,
    LocationSensor,
    LeaderSensor,
    ResourceSensor,
    SuccessSensor,
    RandomGenerator,
    DistanceSensor
)
fun Aggregate<Int>.spawnAndDestroyAfterStability(): Double = with(this@DistanceSensor) {
    vmc { potential: Double, localSuccess: Double, success: Double, localResource: Double ->
        val children = neighboring(findParent(potential))
        set("children-around", children)
        set("myParent", children.localValue)
        val childrenCount = children
            .fold(0) { acc, parent -> acc + if (parent == localId) 1 else 0 }
        set("children-count", childrenCount)
        val neighbors = neighboring(coordinates())
        val localPosition = neighbors.localValue
        val neighborPositions = surroundings()
        val now = currentTime()
        data class Stability(val spawnStable: Boolean = false, val destroyStable: Boolean = false) {
            infix fun and(other: Stability): Boolean = spawnStable && other.spawnStable && destroyStable && other.destroyStable
        }
        share(Stability()) { neighborhoodStability ->
            val lastChanged = evolve(now to listOf(potential, localSuccess, success, localResource)) { last ->
                val current = listOf(potential, localSuccess, success, localResource)
                if (current == last.second) { last } else { now to current }
            }.first
            val enoughTime = now > lastChanged + minSpawnWait
            val everyoneIsStable = neighborhoodStability.fold(enoughTime) { acc, change -> acc and change.destroyStable && change.spawnStable }
            val everyoneIsDestroyStable = neighborhoodStability.fold(lastChanged != now) { acc, change -> acc and change.destroyStable }
            set("enough-time", enoughTime)
            set("everyone-is-stable", everyoneIsStable)
            set("everyone-is-destroy-stable", everyoneIsDestroyStable)
            val localStability = neighborhoodStability.localValue
            when {
                potential > 0.0 && childrenCount == 0 && localResource < resourceLowerBound && everyoneIsDestroyStable -> {
                    selfDestroy()
                    Stability(spawnStable = false, destroyStable = false)
                }
                neighborPositions.isEmpty() || localResource / (2 + childrenCount) > resourceLowerBound && childrenCount < maxChildren && everyoneIsStable -> {
                    val relativePositions = neighborPositions.map { it - localPosition }
                    val angles = relativePositions.map { atan2(it.second, it.first) }.sorted()
                    fun relativeAngleTowards(center: Double) = PI * nextGaussian() / maxChildren + center
                    val angle = when {
                        angles.isEmpty() -> nextRandomDouble(0.0..2 * PI)
                        angles.size == 1 -> relativeAngleTowards(angles.first() + PI)
                        else -> {
                            val fullCircle = angles + (angles.first() + 2 * PI)
                            data class Angle(val from: Double, val arc: Double) : Comparable<Angle> {
                                override fun compareTo(other: Angle): Int =
                                    compareBy(Angle::arc).thenBy(Angle::from).compare(this, other)
                            }
                            val minArc = 2 * PI / maxChildren
                            val differences = fullCircle
                                .zipWithNext { a, b -> Angle(a, b - a) }
                                .filter { it.arc >= minArc }
                            when {
                                differences.isEmpty() -> Double.NaN
                                else -> {
                                    val selectedAngle = differences.randomElementWeighted { arc }
                                    relativeAngleTowards(selectedAngle.arc / 2 + selectedAngle.from)
                                }
                            }
                        }
                    }
                    when {
                        angle.isNaN() -> Stability(spawnStable = true, destroyStable = true)
                        else -> {
                            val x = cloningRange * cos(angle)
                            val y = cloningRange * sin(angle)
                            val absoluteDestination = localPosition + (x to y)
                            spawn(absoluteDestination)
                            Stability(spawnStable = false, destroyStable = localStability.destroyStable)
                        }
                    }
                }
                else -> Stability(spawnStable = enoughTime, destroyStable = now > lastChanged)
            }
        }
    }
}

/**
 * The VMC algorithm with the spawning and destroying of nodes.
 * First it elects the leader, then it calculates the potential,
 * the local success, and the overall success of the children.
 * Finally, it calculates the local resource and checks the spawn and destroy policies.
 */
context(
    EnvironmentVariables,
    DistanceSensor,
    DeviceSpawn,
    LocationSensor,
    LeaderSensor,
    ResourceSensor,
    SuccessSensor
)
fun Aggregate<Int>.vmc(spawner: Spawner): Double {
    val isLeader = isLeader()
    val potential = findPotential(isLeader)
    val localSuccess = obtainLocalSuccess()
    val success = convergeSuccess(potential, localSuccess)
    val localResource = spreadResource(potential, success)
    spawner(this@DeviceSpawn, this@LocationSensor, this, potential, localSuccess, success, localResource)
    return localResource
}

typealias Spawner = context(DeviceSpawn, LocationSensor) Aggregate<Int>.(potential: Double, localSuccess: Double, success: Double, localResource: Double) -> Unit
