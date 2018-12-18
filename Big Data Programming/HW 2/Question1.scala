object Question1 extends App {

	println("The factorial of 5 is " + f(5))
	println("The factorial of 8 is " + f(8))

	def f (n: Int): Int = if (n < 1) 1 else (n to 1 by -1).reduceLeft(_*_)
}
