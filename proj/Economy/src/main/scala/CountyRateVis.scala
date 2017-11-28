import java.io.File

import Unemployment.{directory, sc}
import org.apache.spark.SparkConf
import org.apache.spark.sql.Encoders
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.trim

import scalafx.application.JFXApp
import org.apache.spark.sql.functions._
import swiftvis2.plotting._
import swiftvis2.plotting.renderer.FXRenderer

//case class Series(sid: String, area: String, measureCode: String, title: String)
case class LACData(id: String, year: Int, period: String, value: Double)
case class ZipData(zipCode: String, lat: Double, lon: Double, city: String, state: String, county: String)
case class ZipCountyData(lat: Double, lon: Double, county: String, state: String)
case class IncomeData(geoID:String, geoName:String, code: Int, pi2006: Double, pi2009: Double, pi2015: Double) //code == 3 -> personal income code == 2 -> population

object CountyRateVis extends JFXApp {
  val conf = new SparkConf().setAppName("BLST").setMaster("local[*]")
  val ss = SparkSession.builder().config(conf).getOrCreate()
  import ss.implicits._
  //import ss.sqlContext.implicits._
  ss.sparkContext.setLogLevel("WARN")

  val directory = "data/"

  //create schema use Encoders and case class
  //use select can choose columns and apply functions from sql.functions to each column as trim or regex
  //use sample function can randomly choose certain % amount of data from the orginal dataset with or without replacement
  val countyData = ss.read.schema(Encoders.product[LACData].schema)
    .option("header", true).option("delimiter", "\t")
    .csv(directory + "la_data_64_County.txt").select(trim('id) as "id", 'year, 'period, 'value)
    .as[LACData].filter('id.endsWith("03")).cache() //for production
  //  .as[LACData].sample(false, 0.1).cache() //for test using sample

  countyData.show()
  val seriesData = ss.read.textFile(directory + "la_series.txt").filter(!_.contains("series_id")).map { x =>
    val p = x.split("\t").map(_.trim())
    Series(p(0), p(2), p(3), p(6))
  }.cache()
  seriesData.show()

  //join vs joinWith:  joinWith is a type-preserving join (return tuples instread of dataframe)
  val joinedOne = countyData.joinWith(seriesData, 'id === 'sid).cache()
  //  joinedOne.show()
//  println(joinedOne.first())

  val zipData = ss.read.schema(Encoders.product[ZipData].schema).option("header", true).csv(directory + "zip_codes_states.csv").as[ZipData].filter('lat.isNotNull).cache()
  //zipData.show()

  val countyLoc = zipData.groupByKey{ x => x.city -> x.state }.agg(avg('lat).as[Double], avg('lon).as[Double]).map{
    case ((county, state), lat, lon) =>
      ZipCountyData(lat, lon, county, state)
  }

  //  countyLoc.show()
  val fullJoined = joinedOne.joinWith(countyLoc, '_2("title").contains('county) && '_2("title").contains('state))
  println(fullJoined.first())
  println(fullJoined.count())

  val values = fullJoined.map(_._1._1.value).collect()
  val lon = fullJoined.map(_._2.lon).collect
  val lat = fullJoined.map(_._2.lat).collect

  val cg = ColorGradient(0.0 -> BlueARGB, 4.0 -> GreenARGB, 8.0 -> RedARGB)
  val plot = Plot.scatterPlot(lon, lat, "", "longtitude", "lattitude", 3, values.map(cg))

  FXRenderer(plot, 800, 600)
  FXRenderer.saveToImage(plot, 800, 600, new File("US_Unemployment_Rate_county.png"))

  //county based personal income data
  val incomeData = ss.read.textFile(directory + "CA1_1969_2015__ALL_AREAS.csv").filter(!_.contains("GeoFIPS")).flatMap{
    line =>
      val v = line.split(",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)", -1).map(_.trim).map(_.replaceAll("\"", ""))
      if(v.length < 5 ||v(0).toInt % 1000 == 0 || v(4).toInt == 1 || v(4).toInt == 2){
        Seq.empty
      }else
        Seq(IncomeData(v(0), v(1), v(4).toInt, v(44).toDouble, v(47).toDouble, v(53).toDouble))
  }

  incomeData.show()

  ss.stop()
}
