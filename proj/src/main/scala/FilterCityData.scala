import org.apache.spark.{SparkConf, SparkContext}

case class Area(code: String, text: String)
case class Series(id: String, area: String, measureCode: String, title: String)
case class LAData(id: String, year: Int, period: Int, value: Double)

object FilterCityData {
  val directory = "data/"

  //create spark context
  //using local machine all cores, if on spark cluster, change the setMaster
  val conf = new SparkConf().setAppName("PreprocessUnemploymentRate").setMaster("local[*]")
  val sc = new SparkContext(conf)
  sc.setLogLevel("WARN")

  //read in Area data from la_area.txt
  val areaData = sc.textFile(directory + "la_area.txt")
                    .filter(!_.contains("area_type_code"))
                    .map{ line =>
                      val v = line.split("\t").map(_.trim)
                      Area(v(1), v(2))
                    }.cache()
  areaData.take(10) foreach println

}
