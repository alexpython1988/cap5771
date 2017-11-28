import java.io.File
import org.apache.spark.{SparkConf, SparkContext}
import swiftvis2.plotting._
import swiftvis2.plotting.renderer.FXRenderer
import scalafx.application.JFXApp

case class Area(code: String, text: String)
case class Series(sid: String, area: String, measureCode: String, title: String)
case class UnemploymentRate(id: String, year: Int, period: Int, value: Double)
case class GeoData(zipCode: String, lat: Double, lon: Double, city: String, state: String, county: String)
case class GeoCityData(lat: Double, lon: Double, city: String, state: String)
case class Temp(x: String, y:String)
case class GDPData(geoID:String, geoName: String, industryID: Int, gdp2006: Double, gdp2009: Double, gdp2015: Double) //only use ID==1 ->all industry
case class IncomePerPersonData(geoID:String, geoName:String, code: Int, po2008: Double, po2009: Double, po2015: Double) //code == 3 -> personal income code == 2 -> population

object Unemployment extends JFXApp{
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
                      Area(v(1), v(2).split(",")(0))
                    }
//  areaData.take(10) foreach println

  //read in Series data from la_series.txt
  //series data has 4 kinds of measures, we only need the measure with code==03
  val seriesData = sc.textFile(directory + "la_series.txt").flatMap{ line =>
    if(line.contains("series_id"))
      Seq.empty
    else{
      val v = line.split("\t").map(_.trim)
      Seq(Series(v(0), v(2), v(3), v(6)))
    }
  }.filter(_.measureCode == "03")
  //seriesData.take(5) foreach println

  //read in city unemployment rate data from la_data_Metro.txt
  val unemploymentRateData = sc.textFile(directory + "la_data_Metro.txt").flatMap{line =>
    if(line.contains("series_id") || line.contains("-"))
      Seq.empty
    else{
      val v = line.split("\t").map(_.trim)
      Seq(UnemploymentRate(v(0), v(1).toInt, v(2).filter(_.isDigit).toInt, v(3).toDouble))
    }
  }.filter(_.id.endsWith("03"))

  //unemploymentRateData.take(5) foreach println

  /*
   *  average each city, each month rate to represent the yearly rate for year from 2005 to 2015
   *  using aggregate and mapValue to obtain average year data for each city and year
   *  return as a map with key as (city, year) and value is yearly average unemployment rate
   */
  val unemploymentRateYearGroup = unemploymentRateData.filter{ urd =>
    urd.year >= 2005 && urd.year <= 2015
  }.map{ urd =>
    (urd.id, urd.year) -> urd.value
  }.aggregateByKey(0.0 -> 0)({
    case ((s, c), v) =>
      (s+v, c+1)
  }, {
    case ((s1, c1), (s2, c2)) =>
      (s1+s2, c1+c2)
  })

//  unemploymentRateYearGroup.take(10) foreach println

  val unemploymentRateYearAvg = unemploymentRateYearGroup.mapValues{
    case (s,c) => s/c
  }.cache()
  //unemploymentRateYearAvg.take(5) foreach println

  /*
      merge unemployment data with city information through series on year 2006 2009 2015
      read in cities' geo location data zip_codes_states.csv (using lat and lon for plotting)
      combine unemployment data with geo location data to visualization the rate ~ location data on three years
      the rate will be represented by color
   */

  val uRYA2006 = unemploymentRateYearAvg.filter(_._1._2 == 2006).map{
    case ((id, y), r) =>
      id -> r
  }
  //uRYA2006.take(10) foreach println
  val uRYA2009 = unemploymentRateYearAvg.filter(_._1._2 == 2009).map{
    case ((id, y), r) =>
      id -> r
  }
  val uRYA2015 = unemploymentRateYearAvg.filter(_._1._2 == 2015).map{
    case ((id, y), r) =>
      id -> r
  }

  val geoData = sc.textFile(directory + "zip_codes_states.csv").flatMap{ line =>
    if(line.contains("zip_code"))
      Seq.empty
    else{
      val v = line.replaceAll("\"", "").split(",")
      if(v(1)=="" || v(2) == "")
        Seq.empty
      else
        Seq(GeoData(v(0), v(1).toDouble, v(2).toDouble, v(3), v(4), v(5)))
    }
  }.cache()
  //geoData.foreach(println)

  val cityLoc = geoData.flatMap{ x =>
    Seq(GeoCityData(x.lat, x.lon, x.city, x.state))
  }.map{
    gcd =>
      gcd.city -> (gcd.lat, gcd.lon)
  }.aggregateByKey((0.0, 0.0) -> 0)({
    case (((la, lo), c), (vla,vlo)) =>
      ((la+vla, lo+vlo), c+1)
  },{
    case (((la1, lo1), c1),((la2, lo2), c2)) =>
    ((la1+la2, lo1+lo2), c1+c2)
  }).mapValues{
    case ((las, los), c) => (las/c, los/c)
  }.cache()
//  cityLoc.take(10).foreach(println)

  val seriesMap = seriesData.map{
    x =>
      x.sid -> (x.area, x.title, x.measureCode)
  }.cache()


  val city2Code = areaData.map{ area =>
    val k = List[String](area.code).toArray
    if(area.text.contains("-")){
      val x = area.text.split("-")
      var l = List[String](x(0))
      for(i <- 1 until x.length){
        l = x(i) :: l
      }
      k -> l.toArray
    }else{
      k -> List[String](area.text).toArray
    }
  }.map{
    x =>
      x._2.flatMap(c => x._1.map(c->)).toMap
  }.flatMap(_.toSeq).map{
    case (x, y) =>
      (y, x)
  }.cache()

//  city2Code.take(10) foreach println
  //TODO uncomment back after finish GDP and INCOME process
  //**********************************************************************************************

//  val cityLocRate2006 = uRYA2006.join(seriesMap).map{
//    case (k, (rate, (code, text, c))) =>
//      code -> rate
//  }.join(city2Code).map {
//    case (k, (r, c)) => c -> r
//  }.join(cityLoc).map{
//    case (c,(r,(lat, lon))) =>
//      (r, lat, lon)
//  }

////  println(cityLocRate2006.count())

//  val lat1 = cityLocRate2006.map(_._2).collect()
//  val lon1 = cityLocRate2006.map(_._3).collect()
//  val rate = cityLocRate2006.map(_._1).collect()
//
//  val cg = ColorGradient(1.5 -> BlueARGB, 4.5 -> GreenARGB, 8.2 -> RedARGB)
//  val plot = Plot.scatterPlot(lon1, lat1, "", "longtitude", "lattitude", 7, rate.map(cg))
//
//  FXRenderer(plot, 800, 600)
////  FXRenderer.saveToImage(plot, 800, 600, new File("US_Unemployment_Rate_2006.png"))
//
//  val cityLocRate2009 = uRYA2009.join(seriesMap).map{
//    case (k, (rate, (code, text, c))) =>
//      code -> rate
//  }.join(city2Code).map {
//    case (k, (r, c)) => c -> r
//  }.join(cityLoc).map{
//    case (c,(r,(lat, lon))) =>
//      (r, lat, lon)
//  }
//
//  val rate1 = cityLocRate2009.map(_._1).collect()
//
//  val plot1 = Plot.scatterPlot(lon1, lat1, "", "longtitude", "lattitude", 7, rate1.map(cg))
//
//  FXRenderer(plot1, 800, 600)
////  FXRenderer.saveToImage(plot1, 800, 600, new File("US_Unemployment_Rate_2009.png"))
//
//  val cityLocRate2015 = uRYA2015.join(seriesMap).map{
//    case (k, (rate, (code, text, c))) =>
//      code -> rate
//  }.join(city2Code).map {
//    case (k, (r, c)) => c -> r
//  }.join(cityLoc).map{
//    case (c,(r,(lat, lon))) =>
//      (r, lat, lon)
//  }
//
//  val rate2 = cityLocRate2015.map(_._1).collect()
//
//  val plot2 = Plot.scatterPlot(lon1, lat1, "", "longtitude", "lattitude", 7, rate2.map(cg))
//
//  FXRenderer(plot2, 800, 600)
////  FXRenderer.saveToImage(plot2, 800, 600, new File("US_Unemployment_Rate_2015.png"))
  //**********************************************************************************************

  //read in GDP data
  //case class GDPData(geoID:String, geoName: String, industryID: Int, d2006: Double, d2009: Double, d2015: Double)
  val gdpData = sc.textFile(directory + "gmpGDP.csv").filter(!_.contains("GeoFIPS")).flatMap{ line =>
    val v = line.split(",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)", -1).map(_.trim).map(_.replaceAll("\"", ""))
    if (v.length < 20 || v(5).toInt != 1)
      Seq.empty
    else
      Seq(GDPData(v(0), v(1), v(5).toInt, v(13).toDouble, v(16).toDouble, v(22).toDouble))
  }.map{
    gdp =>
      gdp.geoID -> (gdp.geoName, gdp.gdp2009, gdp.gdp2015)
  }
//  .take(10) foreach println

  //read in personal income data
  val incomeData = sc.textFile(directory + "RPI_2008_2015_MSA.csv")
    .filter{x => !x.contains("GeoFIPS") && !x.contains("(NA)")}
    .flatMap{
      line =>
        val v = line.split(",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)", -1).map(_.trim).map(_.replaceAll("\"", ""))
        if(v.length < 5 || v(0).toInt == 999 || v(0).toInt == 0 || v(4).toInt == 1 ){
          Seq.empty
        }else
          Seq(IncomePerPersonData(v(0), v(1), v(4).toInt, v(7).toDouble, v(8).toDouble, v(14).toDouble))
    }.map{
      icd =>
        icd.geoID -> (icd.geoName, icd.po2009, icd.po2015)
    }
//  .take(10) foreach println

  //process gdp and income data; join two data together based on city since the data share the same location id
  //only get the data for 2009 and 2015 since 2006 data is not available
  val gdpIncomeData = incomeData.join(gdpData).map{
    case (k, ((c1, pic2009, pic2015), (c2, gdp2009, gdp2015))) =>
      val c3 = c1.split("\\(").map(_.trim)
      c3(0) -> (pic2009, pic2015, gdp2009, gdp2015)
  }.map{ each =>
    val v = List[(Double, Double, Double, Double)](each._2).toArray
    if(each._1.contains("-")){
      val dd = each._1.split(",")
      val c = dd(0).split("-")
      var l = List[String](c(0) + "," + dd(1))
      for(j <- 1 until c.length){
        val t = c(j) + "," + dd(1)
        l = t :: l
      }
      l.toArray -> v
    }else{
      List[String](each._1).toArray -> v
    }
  }.map{ x=>
    x._1.flatMap(c => x._2.map(c->)).toMap
  }.flatMap(_.toSeq).map{
    case (k, (a,b,c,d))=>
      val temp = k.split(",").map(_.trim)
      (temp(0), temp(1)) -> (a,b,c,d)
  } //(Beaufort,SC,(41640.0,45536.0,7541.0,8646.0)) (pi2009, pi2015, gdp2009, gdp2015)
//      .take(10) foreach println
  /*
  ***************************************************************************************
  * process data for plotting personal income and gdp vs location
  */

  val cityLoc1 = geoData.flatMap{ x =>
    Seq(GeoCityData(x.lat, x.lon, x.city, x.state))
  }.map{
    gcd =>
      (gcd.city, gcd.state) -> (gcd.lat, gcd.lon)
  }.aggregateByKey((0.0, 0.0) -> 0)({
    case (((la, lo), c), (vla,vlo)) =>
      ((la+vla, lo+vlo), c+1)
  },{
    case (((la1, lo1), c1),((la2, lo2), c2)) =>
      ((la1+la2, lo1+lo2), c1+c2)
  }).mapValues{
    case ((las, los), c) => (las/c, los/c)
  }
//    .take(10) foreach println

  val dgpIncomeYearLocData =  gdpIncomeData.join(cityLoc1).map{
    case ((c,s), ((pi2009, pi2015, gdp2009, gdp2015),(lat, lon))) =>
      (c, s) -> (pi2009, pi2015, gdp2009, gdp2015, lat, lon)
  }.cache() //(city, state, pi2009, pi2015, gdp2009, gdp2015, lat, lo )
//      .take(10) foreach println //((Flagstaff,AZ),(34414.0,38605.0,4643.0,5573.0,35.687109400000004,-111.8158974))

  //TODO uncomment after finish all the code
  /*
  *****************************************************************************************************************
    plotting data to represent changes on income and gdp in 2009 and 2015 (total 4 figures)

   */
//  val lat2 = dgpIncomeYearLocData.map(_._2._5).collect()
//  val lon2 = dgpIncomeYearLocData.map(_._2._6).collect()
//  //TODO tune RGB param to get better distinguish representation curr_best: 1000, 12000, 500000
//  val cgGDP = ColorGradient(1000.0 -> BlueARGB, 12000.0 -> GreenARGB, 500000.0 -> RedARGB)
//
//  //2009 gdp
//  val gdp2009 = dgpIncomeYearLocData.map(_._2._3).collect()
////  println(gdp2009.max) //741630.0
////  println(gdp2009.min) //1876.0
//  val plot3 = Plot.scatterPlot(lon2, lat2, "", "longtitude", "lattitude", 10, gdp2009.map(cgGDP))
//  FXRenderer(plot3, 800, 600)
//  FXRenderer.saveToImage(plot3, 800, 600, new File("US_GDP_CITY_2009.png"))
//
//  //2016 gdp
//  val gdp2015 = dgpIncomeYearLocData.map(_._2._4).collect()
//  val plot4 = Plot.scatterPlot(lon2, lat2, "", "longtitude", "lattitude", 10, gdp2015.map(cgGDP))
//  FXRenderer(plot4, 800, 600)
//  FXRenderer.saveToImage(plot4, 800, 600, new File("US_GDP_CITY_2015.png"))
//
//  //income
//  //TODO tune param, cur_best:
//  val cgIncome = ColorGradient(25000.0 -> BlueARGB, 37000.0 -> GreenARGB, 68000.0 -> RedARGB)
//
//  //2009 income
//  val inc2009 = dgpIncomeYearLocData.map(_._2._1).collect()
////  println(inc2009.max) //82233
////  println(inc2009.min) //25472
//
//  val plot5 = Plot.scatterPlot(lon2, lat2, "", "longtitude", "lattitude", 10, inc2009.map(cgIncome))
//  FXRenderer(plot5, 800, 600)
//  FXRenderer.saveToImage(plot5, 800, 600, new File("US_PERSONALINCOME_CITY_2009.png"))
//  //2016 income
//  val inc2015 = dgpIncomeYearLocData.map(_._2._2).collect()
//  val plot6 = Plot.scatterPlot(lon2, lat2, "", "longtitude", "lattitude", 10, inc2015.map(cgIncome))
//  FXRenderer(plot6, 800, 600)
//  FXRenderer.saveToImage(plot6, 800, 600, new File("US_PERSONALINCOME_CITY_2015.png"))
//*****************************************************************************************************************

  //merge unemployment rate with GDP and income data and output



  //terminate spark
  sc.stop()
  println("done")
//  Thread.sleep(3000)
//  System.exit(0)
}
