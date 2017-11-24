name := "proj"

version := "0.1"

scalaVersion := "2.11.8"

libraryDependencies += "org.scalafx" % "scalafx_2.11" % "8.0.102-R11"
libraryDependencies += "org.apache.spark" % "spark-core_2.11" % "2.2.0"
libraryDependencies += "org.apache.spark" % "spark-sql_2.11" % "2.2.0"
libraryDependencies += "org.apache.spark" % "spark-mllib_2.11" % "2.2.0"
libraryDependencies += "org.apache.spark" % "spark-graphx_2.11" % "2.2.0"
libraryDependencies += "org.scala-lang" % "scala-library" % scalaVersion.value
libraryDependencies += "org.scala-lang" % "scala-compiler" % scalaVersion.value
libraryDependencies += "org.scala-lang" % "scala-reflect" % scalaVersion.value