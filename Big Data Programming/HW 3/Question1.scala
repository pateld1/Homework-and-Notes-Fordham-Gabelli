val file = sc.textFile("alice.txt")
val counts = file.flatMap(line => line.split(" ")).map(word => (word, 1)).reduceByKey(_+_)
counts.saveAsTextFile("output")