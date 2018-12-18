object Question4 extends App{

	import scala.io.Source
	import java.io._

	reverseFile("alice.txt", "rev.txt")

	def reverseFile(input: String, output: String){
		println("File is being read.")
		val file = Source.fromFile(input)
		val lines = file.getLines.toArray
		val rev = lines.reverse 
		val writer = new PrintWriter(new File(output))
		println("File is being written.")
		rev.foreach(writer.write)
		writer.close()
	}
}
