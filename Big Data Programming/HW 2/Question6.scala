object Question6 extends App{
	
	println("The LCM of 10 and 49 is: " + lcm(10, 40))
	println("The LCM of 65 and 30 is: " + lcm(65, 30))
	println("The LCM of 3 and 5 is: " + lcm(3,5))
	println("The LCM of 6 and 3 is: " + lcm(6, 3))
	println("The LCM of 12 and 48 is: " + lcm(12, 48))

	def gcd(a: Int, b: Int): Int = {
		if(b == 0) a
		else gcd(b, a % b)
	}

	def lcm(a: Int, b: Int): Int = (a * b) / gcd(a, b)
}