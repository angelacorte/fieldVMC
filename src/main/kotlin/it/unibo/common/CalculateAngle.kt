package it.unibo.common

import it.unibo.collektive.alchemist.device.sensors.RandomGenerator
import kotlin.math.PI

/**
 * Selects a [random] element from the iterable,
 * with a probability proportional to the value returned by the [by] function.
 */
fun <T> Iterable<T>.randomElementWeighted(
    random: RandomGenerator,
    by: T.() -> Double,
): T {
    val total = fold(0.0) { acc, element -> acc + by(element) }
    val selector: Double = random.nextRandomDouble(0.0..total)
    var accumulator = 0.0
    for (element in this) {
        accumulator += by(element)
        if (accumulator >= selector) {
            return element
        }
    }
    return last()
}

/**
 * Selects a random angle from the list of [angles],
 * with a probability proportional to the angle's distance from the previous and next angle.
 * The [maxChildren] parameter is used to calculate the minimum angle difference.
 */
fun calculateAngle(
    angles: List<Double>,
    random: RandomGenerator,
    maxChildren: Int,
): Double {
    fun relativeAngleTowards(center: Double) = PI * random.nextGaussian() / maxChildren + center
    return when {
        angles.isEmpty() -> random.nextRandomDouble(0.0..2 * PI)
        angles.size == 1 -> relativeAngleTowards(angles.first() + PI)
        else -> {
            val fullCircle = angles + (angles.first() + 2 * PI)

            data class Angle(
                val from: Double,
                val arc: Double,
            ) : Comparable<Angle> {
                override fun compareTo(other: Angle): Int = compareBy(Angle::arc).thenBy(Angle::from).compare(this, other)
            }
            val minArc = 2 * PI / maxChildren
            val differences =
                fullCircle
                    .zipWithNext { a, b -> Angle(a, b - a) }
                    .filter { it.arc >= minArc }
            when {
                differences.isEmpty() -> Double.NaN
                else -> {
                    val selectedAngle = differences.randomElementWeighted(random) { arc }
                    relativeAngleTowards(selectedAngle.arc / 2 + selectedAngle.from)
                }
            }
        }
    }
}
