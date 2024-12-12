package it.unibo.collektive.alchemist.device.sensors

interface SuccessSensor {

    /**
     * Set the [success] accumulated at the current node.
     */
    fun setSuccess(success: Double)

    /**
     * Set the [localSuccess] for the current node.
     */
    fun setLocalSuccess(localSuccess: Double)

    /**
     * Returns the success accumulated at the current node.
     */
    fun getSuccess(): Double

    /**
     * Returns the local success at the current node, sensing the success layer.
     */
    fun getLocalSuccess(): Double
}
