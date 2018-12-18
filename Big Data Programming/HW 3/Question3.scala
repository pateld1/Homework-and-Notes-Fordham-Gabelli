import org.apache.spark.sql.types._
import org.apache.spark.sql.Row

def moneyToFloat (money: String): Float = {
	money.replace("$", "").replace(",","").toFloat
}

def transformToRow (input: (String, (Float, Float))): Row = {
	Row(input._1, input._2._1, input._2._2)
}

val schema = 
	StructType (
		StructField("Department", StringType, false)::
		StructField("MinimumSalary", FloatType, false)::
		StructField("MaximumSalary", FloatType, false) :: Nil)
	
val empsal = sc.textFile("empsal.txt")
val getMaxs = empsal.map(lines=>lines.split('\t')).map(x=>(x(1), moneyToFloat(x(5)))).reduceByKey(math.max(_,_))
val getMins = empsal.map(lines=>lines.split('\t')).map(x=>(x(1), moneyToFloat(x(5)))).reduceByKey(math.min(_,_))
val sortMaxs = getMaxs.sortByKey()
val sortMins = getMins.sortByKey()
val joined = sortMins.join(sortMaxs)
val together = joined.map(transformToRow)
val df = spark.createDataFrame(together, schema)

df.coalesce(1).write.format("com.databricks.spark.csv").option("header", "true").save("deptMinMaxSalary.csv")


