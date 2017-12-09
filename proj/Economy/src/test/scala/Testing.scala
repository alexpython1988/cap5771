import org.apache.spark.{SparkConf, SparkContext}

object Testing extends App{
  val conf = new SparkConf().setAppName("PreprocessUnemploymentRate").setMaster("local[*]")
  val sc = new SparkContext(conf)
  sc.setLogLevel("WARN")
  val input = sc.parallelize((1 to 100).toArray.toSeq)
  val input1 = input.zip(sc.parallelize((1 to input.count().toInt).toArray.toSeq))
  input1 foreach println

  //filter
  input.filter(x => x % 2 != 0) foreach println

  //filter with implementation
  input.flatMap{
    x =>
      if (x % 2 != 0)
        Seq(x)
      else
        Seq.empty
  } foreach println

  //count
  println(input.count())

  //use map reduce
  val c = input.map(x => 1).reduce(_ + _)
  println(c)




}
