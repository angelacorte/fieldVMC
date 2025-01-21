@file:Suppress("DEPRECATION")

package it.unibo.alchemist.boundary.swingui.effect.impl

import it.unibo.alchemist.boundary.swingui.effect.api.Effect
import it.unibo.alchemist.boundary.swingui.effect.impl.DrawTree.Companion.circlesThrough
import it.unibo.alchemist.boundary.swingui.effect.impl.DrawTreeNodes.Companion.localSuccess
import it.unibo.alchemist.boundary.swingui.effect.impl.DrawTreeNodes.Companion.resource
import it.unibo.alchemist.boundary.ui.api.Wormhole2D
import it.unibo.alchemist.model.Environment
import it.unibo.alchemist.model.Node
import it.unibo.alchemist.model.Position2D
import it.unibo.alchemist.model.Time
import it.unibo.alchemist.model.molecules.SimpleMolecule
import org.apache.commons.math3.util.FastMath.pow
import java.awt.BasicStroke
import java.awt.Color
import java.awt.Graphics2D
import java.awt.Point
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.sin
import kotlin.math.sqrt

class DrawTree : Effect {
    override fun getColorSummary(): Color = Color.BLACK

    @Transient
    var lastUpdated = Time.NEGATIVE_INFINITY

    @Transient
    var maxSuccess = 0.0

    @Transient
    var maxResource = 0.0

    override fun <T : Any?, P : Position2D<P>> apply(
        g: Graphics2D,
        node: Node<T>,
        environment: Environment<T, P>,
        wormhole: Wormhole2D<P>,
    ) {
        if (environment.simulation.time != lastUpdated) {
            maxSuccess = 0.0
            maxResource = 0.0
            for (currentNode in environment.nodes) {
                maxSuccess = max(maxSuccess, currentNode.getConcentration(localSuccess).toDouble())
                maxResource = max(maxResource, currentNode.getConcentration(resource).toDouble())
            }
            lastUpdated = environment.simulation.time
        }
        val parentId: Int? = node.getConcentration(myParent).toInt()
        val localResource = node.getConcentration(resource).toDouble()
        val localSuccess = node.getConcentration(localSuccess).toDouble()
        runCatching {
            environment.getNeighborhood(node)
        }.onSuccess { neighborhood ->
            val localPosition = environment.getPosition(node)
            val viewPoint: Point = wormhole.getViewPoint(localPosition)
            val shifted: P = environment.makePosition(0.3, 0.3) + localPosition.coordinates
            val shiftedViewPoint = wormhole.getViewPoint(shifted)
            val lineDistance = abs(shiftedViewPoint.x - viewPoint.x)
            for (neighbor in neighborhood) {
                runCatching { environment.getPosition(neighbor) }.onSuccess { neighborPosition ->
                    val screenNeighborPosition = wormhole.getViewPoint(neighborPosition)
                    if (neighbor.id == parentId) {
                        val (midpoint1, midpoint2) = midpointShift(viewPoint, screenNeighborPosition, lineDistance)
                        g.stroke = BasicStroke(lineDistance * (localSuccess / maxSuccess).toFloat(), BasicStroke.CAP_BUTT, BasicStroke.JOIN_MITER)
                        g.color = Color(234, 109, 0)
                        fun drawLine(p1: Point, p2: Point) = g.drawLine(p1.x, p1.y, p2.x, p2.y)
                        drawLine(viewPoint, midpoint1)
                        drawLine(midpoint1, screenNeighborPosition)
                        //                g.drawString("%.2f".format(localSuccess), midpoint1.x, midpoint1.y)
                        g.stroke = BasicStroke(2 * lineDistance * (localResource / maxResource).toFloat(), BasicStroke.CAP_BUTT, BasicStroke.JOIN_MITER)
                        g.color = Color(85, 170, 0)
                        drawLine(viewPoint, midpoint2)
                        drawLine(midpoint2, screenNeighborPosition)
                        //                g.drawString("%.2f".format(localResource), midpoint2.x, midpoint2.y)
                    }
                    g.color = Color.BLACK
                    g.stroke = when {
                        neighbor.id == parentId -> BasicStroke(4f)
                        else -> BasicStroke(1.5f, BasicStroke.CAP_BUTT, BasicStroke.JOIN_MITER, 10f, dashPattern, 0f)
                    }
                    g.drawLine(viewPoint.x, viewPoint.y, screenNeighborPosition.x, screenNeighborPosition.y)
                }
            }
        }
    }

    companion object {
        val myParent = SimpleMolecule("myParent")
        val dashPattern = floatArrayOf(1f, 5f)
        val minArcDistance = 10

        private fun Any?.toInt(): Int? = when (this) {
            is Int -> this
            is Number -> this.toInt()
            is String -> this.toInt()
            null -> null
            Unit -> null
            else -> error("Unexpected integer: $this")
        }

        private fun Any?.toDouble(): Double = when (this) {
            is Double -> this
            is Number -> this.toDouble()
            null -> 0.0
            Unit -> 0.0
            else -> error("Unexpected integer: $this")
        }

        private operator fun Point.plus(other: Point): Point = Point(x + other.x, y + other.y)
        private operator fun Point.minus(other: Point): Point = Point(x - other.x, y - other.y)

        data class Line(val x: Int, val y: Int, val slope: Double)

        fun midpoint(p1: Point, p2: Point): Point {
            return Point((p1.x + p2.x) / 2, (p1.y + p2.y) / 2)
        }

        fun perpendicularBisector(p1: Point, p2: Point): Line {
            val midPoint = midpoint(p1, p2)
            val slope = if (p2.y != p1.y) {
                -(p2.x.toDouble() - p1.x.toDouble()) / (p2.y.toDouble() - p1.y.toDouble())
            } else { 0.0 }
            val bisectorSlope = if (slope != 0.0) -1 / slope else Double.POSITIVE_INFINITY
            return Line(midPoint.x, midPoint.y, bisectorSlope)
        }

        fun circleIntersection(r: Double, x1: Int, y1: Int, x2: Int, y2: Int): Array<Point> {
            val d = hypot((x2 - x1).toDouble(), (y2 - y1).toDouble())
            check(d < r)
            val h = hypot(r, d)
            val xc1 = x2 + h * (y1 - y2) / d
            val yc1 = y2 + h * (x2 - x1) / d
            val xc2 = x2 - h * (y1 - y2) / d
            val yc2 = y2 - h * (x2 - x1) / d
            return arrayOf(Point(xc1.toInt(), yc1.toInt()), Point(xc2.toInt(), yc2.toInt()))
        }

        fun midpointShift(p1: Point, p2: Point, d: Int): Pair<Point, Point> {
            val midpoint = midpoint(p1, p2)
            val angle = atan2((p2.y - p1.y).toDouble(), (p2.x - p1.x).toDouble())
            val points = listOf(Math.PI / 2, 3 * Math.PI / 2).map { deltaAngle ->
                val p3x = midpoint.x + d * cos(angle + deltaAngle)
                val p3y = midpoint.y + d * sin(angle + deltaAngle)
                Point(p3x.toInt(), p3y.toInt())
            }.sortedBy { it.y }
            return points[0] to points[1]
        }

        fun circlesThrough(p1: Point, p2: Point, d: Int): List<Circle> {
            require(d > 0)
            val midpoint = midpoint(p1, p2)
            val angle = atan2((p2.y - p1.y).toDouble(), (p2.x - p1.x).toDouble())
            return listOf(Math.PI / 2, 3 * Math.PI / 2).map { deltaAngle ->
                val p3x = midpoint.x + d * cos(angle + deltaAngle)
                val p3y = midpoint.y + d * sin(angle + deltaAngle)
                val p3 = Point(p3x.toInt(), p3y.toInt())
                // Check if the points are not collinear
                check((p2.y - p1.y) * (p3.x - p2.x) != (p3.y - p2.y) * (p2.x - p1.x)) {
                    println("Points are collinear, can't form a unique circle.")
                }
                val offset = p2.x * p2.x + p2.y * p2.y
                val bc = (p1.x * p1.x + p1.y * p1.y - offset) / 2.0
                val cd = (offset - p3.x * p3.x - p3.y * p3.y) / 2.0
                val det = (p1.x - p2.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p2.y)
                val inverseDet = 1 / det.toDouble()
                val centerX = (bc * (p2.y - p3.y) - cd * (p1.y - p2.y)) * inverseDet
                val centerY = (cd * (p1.x - p2.x) - bc * (p2.x - p3.x)) * inverseDet
                val radius = sqrt(pow(p2.x - centerX, 2) + pow(p2.y - centerY, 2))
                Circle(centerX.toInt(), centerY.toInt(), radius)
            }
        }
    }
}

data class Circle(val x: Int, val y: Int, val radius: Double)

fun main() {
    println(circlesThrough(Point(0, 0), Point(0, 10), 15))
}
