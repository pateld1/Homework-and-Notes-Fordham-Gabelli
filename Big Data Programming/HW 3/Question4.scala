val sqlcontext = new org.apache.spark.sql.SQLContext(sc)
val airports = sqlcontext.read.format("csv").option("header", "true").option("inferSchema", "true").load("airports.csv")
airports.createOrReplaceTempView("df")

airports.printSchema()

val q2 = spark.sql("SELECT AirportID FROM df WHERE Latitude < 0 AND Longitude > 0")
q2.show

val q3 = spark.sql("SELECT Country, COUNT(*) City FROM df GROUP BY Country ORDER BY Country")
q3.show

val q4 = spark.sql("SELECT Country, AVG(Altitude) FROM df GROUP BY Country ORDER BY Country")
q4.show

val q5 = spark.sql("SELECT Timezone, COUNT(*) Airport FROM df GROUP BY Timezone ORDER BY Timezone")
q5.show

val q6 = spark.sql("SELECT Country, AVG(Latitude), AVG(Longitude) FROM df GROUP BY Country ORDER BY Country")
q6.show

val q7 = spark.sql("SELECT COUNT(DISTINCT DST) FROM df")
q7.show



