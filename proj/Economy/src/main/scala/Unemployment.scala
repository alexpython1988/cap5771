import java.io.File

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{LogisticRegression, MultilayerPerceptronClassifier, OneVsRest, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.functions.rand
import org.apache.spark.ml.feature.{IndexToString, LabeledPoint, StringIndexer, VectorIndexer}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql._
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
case class GDPData(geoID:String, geoName: String, industryID: Int, gdp2006: Double, gdp2008: Double, gdp2009: Double, gdp2014: Double, gdp2015: Double) //only use ID==1 ->all industry
case class IncomePerPersonData(geoID:String, geoName:String, code: Int, po2008: Double, po2009: Double, po2014:Double, po2015: Double) //code == 3 -> personal income code == 2 -> population

object Unemployment extends JFXApp{
  val directory = "data/"

  //create spark context
  //using local machine all cores, if on spark cluster, change the setMaster
  val conf = new SparkConf().setAppName("PreprocessUnemploymentRate").setMaster("local[*]")
  val ss = SparkSession.builder().config(conf).getOrCreate()
  import ss.implicits._
  val sc = ss.sparkContext
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
  }.filter(_.measureCode == "03").cache()
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
  val uRYA2014 = unemploymentRateYearAvg.filter(_._1._2 == 2014).map{
    case ((id, y), r) =>
      id -> r
  }
  val uRYA2008 = unemploymentRateYearAvg.filter(_._1._2 == 2008).map{
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
  //case class GDPData(geoID:String, geoName: String, industryID: Int, d2006: Double, d2009: Double, 2010, 2014 2015: Double)
  val gdpData = sc.textFile(directory + "gmpGDP.csv").filter(!_.contains("GeoFIPS")).flatMap{ line =>
    val v = line.split(",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)", -1).map(_.trim).map(_.replaceAll("\"", ""))
    if (v.length < 20 || v(5).toInt != 1)
      Seq.empty
    else
      Seq(GDPData(v(0), v(1), v(5).toInt, v(13).toDouble, v(15).toDouble, v(16).toDouble, v(21).toDouble, v(22).toDouble))
  }.map{
    gdp =>
      gdp.geoID -> (gdp.geoName, gdp.gdp2008, gdp.gdp2009, gdp.gdp2014, gdp.gdp2015)
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
          Seq(IncomePerPersonData(v(0), v(1), v(4).toInt, v(7).toDouble, v(8).toDouble, v(13).toDouble, v(14).toDouble))
    }.map{
      icd =>
        icd.geoID -> (icd.geoName, icd.po2008, icd.po2009, icd.po2014, icd.po2015)
    }
//  .take(10) foreach println

  //process gdp and income data; join two data together based on city since the data share the same location id
  //only get the data for 2009 and 2015 since 2006 data is not available
  val gdpIncomeData = incomeData.join(gdpData).map{
    case (k, ((c1, pic2008, pic2009, pic2014, pic2015), (c2, gdp2008, gdp2009, gdp2014, gdp2015))) =>
      val c3 = c1.split("\\(").map(_.trim)
      c3(0) -> (pic2008, pic2009, pic2014, pic2015, gdp2008, gdp2009, gdp2014, gdp2015)
  }.map{ each =>
    val v = List[(Double, Double, Double, Double, Double, Double, Double, Double)](each._2).toArray
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
    case (k, (a,b,c,d,e,f,g,h))=>
      val temp = k.split(",").map(_.trim)
      (temp(0), temp(1)) -> (a,b,c,d,e,f,g,h)
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
    case ((c,s), ((pi2008, pi2009, pi2014, pi2015, gdp2008, gdp2009, gdp2014, gdp2015),(lat, lon))) =>
      (c, s) -> (pi2008, pi2009, pi2014, pi2015, gdp2008, gdp2009, gdp2014, gdp2015, lat, lon)
  }.cache() //(city, state, pi2009, pi2015, gdp2009, gdp2015, lat, lo )
//      .take(10) foreach println //((Flagstaff,AZ),(34414.0,38605.0,4643.0,5573.0,35.687109400000004,-111.8158974))

  //TODO uncomment after finish all the code, need to change param index
  /*
  *****************************************************************************************************************
    plotting data to represent changes on income and gdp in 2009 and 2015 (total 4 figures)

   */
//  val lat2 = dgpIncomeYearLocData.map(_._2._5).collect()
//  val lon2 = dgpIncomeYearLocData.map(_._2._6).collect()
////  //TODO tune RGB param to get better distinguish representation curr_best: 1000, 12000, 500000
//  val cgGDP = ColorGradient(1000.0 -> BlueARGB, 12000.0 -> GreenARGB, 500000.0 -> RedARGB)
////
////  //2009 gdp
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
  val seriesMap1 = seriesData.map{
    x =>
      x.sid -> x.title
  }

  //TODO check order of data after join
  val merged060915 = uRYA2008.join(uRYA2009).join(uRYA2014).join(uRYA2015).map{
    case (k, (((d1, d2), d3), d4)) =>
      k -> (d2, d1, d3, d4) //2008, 2009, 2014, 2015
  }

  val geoUnemployment = merged060915.join(seriesMap1).map{
    case (k,((d08, d09, d14, d15), t)) =>
      (t, d08, d09, d14, d15)
  }

  val employmentDF = geoUnemployment.toDF("info", "rate08", "rate09", "rate14", "rate15")
  val gdpIncomeLocDF = dgpIncomeYearLocData.map{
    case ((k1, k2), (d1,d2,d3,d4,d5,d6,d7,d8,l1,l2)) =>
      (k1,k2,d1,d2,d3,d4,d5,d6,d7,d8,l1,l2)
  }.toDF("city", "state", "pi2008", "pi2009", "pi2014", "pi2015", "gdp2008", "gdp2009", "gdp2014", "gdp2015", "lat", "lon")

  val dataForML = employmentDF.joinWith(gdpIncomeLocDF, 'info.contains('state) && 'info.contains('city)).rdd.map{
    case (Row(t:String, d1: Double, d2:Double, d3:Double, d4:Double), Row(k1:String, k2:String, dd1:Double, dd2:Double, dd3:Double, dd4:Double, dd5:Double, dd6:Double, dd7:Double, dd8:Double, l1:Double, l2:Double)) =>
      (k1,k2,d1,d2,d3,d4,dd1,dd2,dd3,dd4,dd5,dd6,dd7,dd8,l1,l2)
  }.cache()

//  val data2csv = dataForML.toDF("city", "state", "rate08", "rate09", "rate14", "rate15", "pi2008","pi2009", "pi2014", "pi2015", "gdp2008", "gdp2009", "gdp2014", "gdp2015", "lat", "lon")
  //save data to csv
//  data2csv
//    .coalesce(1)
//    .write.format("com.databricks.spark.csv")
//    .option("header", "true")
//    .save("mydata.csv")

//  println(dataForML.count()) //466
  /*
    ML with spark ml
    parallel training against traditional batch-to-batch train
   */
  //example
//  val training = ss.read.format("libsvm")
//    .load("example.txt")

  //format data
  val predict2009Data = dataForML.map{
    case (k1,k2,d1,d2,d3,d4,dd1,dd2,dd3,dd4,dd5,dd6,dd7,dd8,l1,l2) =>
      LabeledPoint(dd6, Vectors.dense(d2, dd2))
  }

  val predict2014Data = dataForML.map{
    case (k1,k2,d1,d2,d3,d4,dd1,dd2,dd3,dd4,dd5,dd6,dd7,dd8,l1,l2) =>
      LabeledPoint(dd7, Vectors.dense(d3, dd3))
  }

  val predict2015Data = dataForML.map{
    case (k1,k2,d1,d2,d3,d4,dd1,dd2,dd3,dd4,dd5,dd6,dd7,dd8,l1,l2) =>
      LabeledPoint(dd8, Vectors.dense(d4, dd4))
  }

  val predict2008Data = dataForML.map{
    case (k1,k2,d1,d2,d3,d4,dd1,dd2,dd3,dd4,dd5,dd6,dd7,dd8,l1,l2) =>
      LabeledPoint(dd5, Vectors.dense(d1, dd1))
  }

  val regressionData = predict2009Data ++ predict2015Data ++ predict2014Data ++ predict2008Data

  //linear regression
  println("******* Linear Regression ***********")
  val lr = new LinearRegression()
    .setMaxIter(100)
    .setRegParam(0.3)
    .setElasticNetParam(0.8)

  val linearModel = lr.fit(regressionData.toDF("label", "features"))
  println(s"Coefficients: ${linearModel.coefficients} Intercept: ${linearModel.intercept}")
  val trainingSummary = linearModel.summary
  println(s"objectiveHistory: [${trainingSummary.objectiveHistory.mkString(",")}]")
  println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
  println(s"r2: ${trainingSummary.r2}")
  println("************************************")
  //plot data as 2D grid
//  val X = regressionData.map(_.features(0)).collect
//  val Y = regressionData.map(_.features(1)).collect
//  val plot7 = Plot.scatterPlot(X, Y, "", "income", "GDP", 3, BlackARGB)
////  FXRenderer(plot7, 800, 600)
//  FXRenderer.saveToImage(plot7, 800, 600, new File("GDP_Income_2D.png"))

  //prepare dataset for classification
  //map state to integer representation
  val states4Key = dataForML.map{
      case (k1,k2,d1,d2,d3,d4,dd1,dd2,dd3,dd4,dd5,dd6,dd7,dd8,l1,l2) =>
        k2 -> 1
  }.reduceByKey(_ + _).filter(_._2 > 2).keys.collect

  val cc = sc.parallelize((1 to states4Key.length).toArray.toSeq).collect
//  println(cc.count())
//  println(states4Key.count())

  val states2Num = states4Key.zip(cc).toMap

  val classificationDataSet2008 = dataForML.filter(x => states2Num.contains(x._2)).map{
    case (k1,k2,d1,d2,d3,d4,dd1,dd2,dd3,dd4,dd5,dd6,dd7,dd8,l1,l2) =>
      LabeledPoint(states2Num.get(k2).get.toDouble, Vectors.dense(d1, dd1, dd5))
  }

  val classificationDataSet2009 = dataForML.filter(x => states2Num.contains(x._2)).map{
    case (k1,k2,d1,d2,d3,d4,dd1,dd2,dd3,dd4,dd5,dd6,dd7,dd8,l1,l2) =>
      LabeledPoint(states2Num.get(k2).get.toDouble, Vectors.dense(d2, dd2, dd6))
  }

  val classificationDataSet2014 = dataForML.filter(x => states2Num.contains(x._2)).map{
    case (k1,k2,d1,d2,d3,d4,dd1,dd2,dd3,dd4,dd5,dd6,dd7,dd8,l1,l2) =>
      LabeledPoint(states2Num.get(k2).get.toDouble, Vectors.dense(d3, dd3, dd7))
  }

  val classificationDataSet2015 = dataForML.filter(x => states2Num.contains(x._2)).map{
    case (k1,k2,d1,d2,d3,d4,dd1,dd2,dd3,dd4,dd5,dd6,dd7,dd8,l1,l2) =>
      LabeledPoint(states2Num.get(k2).get.toDouble, Vectors.dense(d4, dd4, dd8))
  }

  val classificationDataSet = classificationDataSet2008 ++ classificationDataSet2009 ++ classificationDataSet2014 ++ classificationDataSet2015
  val classificationDF = classificationDataSet.toDF("label", "features").orderBy(rand()).cache()

  val Array(ts, tt) = classificationDF.randomSplit(Array(0.8, 0.2))
  val trainset = ts.toDF("label", "features").cache()
  val testset = tt.toDF("label", "features").cache()

  val labelIndexer = new StringIndexer()
    .setInputCol("label")
    .setOutputCol("indexedLabel")
    .fit(classificationDF)

  val featureIndexer = new VectorIndexer()
    .setInputCol("features")
    .setOutputCol("indexedFeatures")
    .setMaxCategories(4)
    .fit(classificationDF)

  val labelConverter = new IndexToString()
    .setInputCol("prediction")
    .setOutputCol("predictedLabel")
    .setLabels(labelIndexer.labels)
  /*
  logistic regression
  using 2009, 2014, 2015, 2008 data to predict state
 */
  //TODO each classifier part can be extracted as a function and passing the classifier as input parameters with p, r, f as returned values
  println("******* Logistic Regression ***********")
  val logr = new LogisticRegression()
    .setMaxIter(100)
    .setRegParam(0.3)
    .setElasticNetParam(0.8)
    .setThreshold(0.5)

  val pipelineLR = new Pipeline()
    .setStages(Array(labelIndexer, featureIndexer, logr, labelConverter))

  val modelLR = pipelineLR.fit(trainset)

  val logrPredictions = modelLR.transform(testset)

  val resForMetricsLR = logrPredictions.select("predictedLabel", "label").rdd.map{
    case (Row(p, l)) =>
      (p.toString.toDouble, l.toString.toDouble)
  }

  val evaluatorLR = new MulticlassClassificationEvaluator()
    .setLabelCol("indexedLabel")
    .setPredictionCol("prediction")
    .setMetricName("accuracy")
  val accuracyLR = evaluatorLR.evaluate(logrPredictions)
  println("Test accuracy = " + accuracyLR)

  val metricsLR = new MulticlassMetrics(resForMetricsLR)
//  println("confusion matrix:")
//  println(metricsLR.confusionMatrix)
  val precisionLR = metricsLR.weightedPrecision
  val recallLR = metricsLR.weightedRecall
  val f1ScoreLR = metricsLR.weightedFMeasure
  println("Summary Statistics")
  println(s"Precision = $precisionLR")
  println(s"Recall = $recallLR")
  println(s"F1 Score = $f1ScoreLR")

  println("**************************************")

  /*
  random forest
   */
  println("***********random forest*******************")

  val rf = new RandomForestClassifier()
    .setLabelCol("indexedLabel")
    .setFeaturesCol("indexedFeatures")
    .setNumTrees(10)

  val pipeline = new Pipeline()
    .setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))

  val model = pipeline.fit(trainset)

  val predictions = model.transform(testset)

  predictions.select("predictedLabel", "label", "features").show(5)

  val evaluator = new MulticlassClassificationEvaluator()
    .setLabelCol("indexedLabel")
    .setPredictionCol("prediction")
    .setMetricName("accuracy")
  val accuracy = evaluator.evaluate(predictions)
  println("Test accuracy = " + accuracy)

  val resForMetricsRF = predictions.select("predictedLabel", "label").rdd.map{
    case (Row(p, l)) =>
      (p.toString.toDouble, l.toString.toDouble)
  }

  val metricsRF = new MulticlassMetrics(resForMetricsRF)
//  println("confusion matrix:")
//  println(metricsRF.confusionMatrix)
  val precisionRF = metricsRF.weightedPrecision
  val recallRF = metricsRF.weightedRecall
  val f1ScoreRF = metricsRF.weightedFMeasure
  println("Summary Statistics")
  println(s"Precision = $precisionRF")
  println(s"Recall = $recallRF")
  println(s"F1 Score = $f1ScoreRF")

  println("***************************************")

  /*
  Multilayer perceptron classifier
   */
  println("*********** Multilayer perceptron classifier *******************")
  val layers = Array[Int](3, 40, 60, 40, 40)
  val mpc = new MultilayerPerceptronClassifier()
    .setLayers(layers)
    .setBlockSize(128)
    .setSeed(1234L)
    .setMaxIter(100)
  val mpcModel = mpc.fit(trainset)
  val predictionsMPC = mpcModel.transform(testset)
  val predictionMPCRes = predictionsMPC.select("prediction", "label")

  val evaluatorMPC = new MulticlassClassificationEvaluator()
    .setMetricName("precision")
  println("Accuracy:" + evaluator.evaluate(predictionMPCRes))

  val metricsMPC = new MulticlassMetrics(predictionMPCRes.rdd.map{
    case (Row(p, l)) =>
      (p.toString.toDouble, l.toString.toDouble)
  })
//  println("confusion matrix:")
//  println(metricsMPC.confusionMatrix)
  val precisionMPC = metricsMPC.weightedPrecision
  val recallMPC = metricsMPC.weightedRecall
  val f1ScoreMPC = metricsMPC.weightedFMeasure
  println("Summary Statistics")
  println(s"Precision = $precisionMPC")
  println(s"Recall = $recallMPC")
  println(s"F1 Score = $f1ScoreMPC")
  println("***************************************")

  //terminate spark
  sc.stop()
  ss.stop()
  println("done")
//  Thread.sleep(3000)
//  System.exit(0)
}
