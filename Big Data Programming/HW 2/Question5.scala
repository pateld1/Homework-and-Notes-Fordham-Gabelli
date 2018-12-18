
object Question5 extends App {

	var x = Array(6,4,8,3,78,46,26,75,13)
	println("Unsorted:")
	x.foreach(println)
	quicksort(x, 0, x.length - 1)
	println("Sorted:")
	x.foreach(println)



	def swap(arr: Array[Int], i : Int, j:Int) {
		val temp = arr(i)
		arr(i) = arr(j)
		arr(j) = temp
	}

	def quicksort(arr: Array[Int], left: Int, right: Int){
		val split = arr((left + right) / 2)
		var i = left
		var j = right

		while(i < j){
			while(arr(i) < split) i += 1
			while(arr(j) > split) j -= 1
			if (i <= j) {
				swap(arr, i,j)
				i += 1
				j -= 1
			}
		}
		if(left < j) quicksort(arr, left, j)
		if(j < right) quicksort(arr, i,right)
	}
}
