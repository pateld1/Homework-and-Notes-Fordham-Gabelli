object Question2 extends App {

	val arr = Array(3,7,4,8,12,13,5,23)
	println("List of Numbers to Test:")
	arr.foreach(println) 
	println("The prime numbers are:")
	for(i <- arr if isPrime(i)) println(i)

	def isPrime(num:Int): Boolean = {
		if(num <= 1) false
		if(num == 2) true 
		List.range(2, num) forall (x => num % x != 0)
	}
}