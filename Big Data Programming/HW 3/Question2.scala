val traveldata = sc.textFile("traveldata.txt")
val splitOne = traveldata.map(lines => lines.split('\t')).map(x=>(x(2),1)).reduceByKey(_+_)
val splitOne_ordered = splitOne.map(x => x.swap).sortByKey(false)
val quesOne = splitOne_ordered.map(x => x.swap).take(20)

val splitTwo = traveldata.map(lines => lines.split('\t')).map(x=>(x(1),1)).reduceByKey(_+_)
val splitTwo_ordered = splitTwo.map(x => x.swap).sortByKey(false)
val quesTwo = splitTwo_ordered.map(x => x.swap).take(20)

val airlinedata = traveldata.map(lines => lines.split('\t')).filter(x => {if ((x(3).matches(("1")))) true else false})
val splitThree = airlinedata.map(x => (x(2), 1)).reduceByKey(_+_)
val splitThree_ordered = splitThree.map(x => x.swap).sortByKey(false)
val quesThree = splitThree_ordered.map(x => x.swap).take(20)

