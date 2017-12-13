object Utils{
  def toDoubleorNeg(s: String): Double = {
    try {
      s.toDouble
    } catch {
      case _: NumberFormatException => -1
    }
  }
}

