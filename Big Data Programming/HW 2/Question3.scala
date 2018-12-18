object Question3 extends App {

	import scala.io.Source
	import java.io._

	reverseLines("alice.txt")

	def reverseLines(f: String){
		println("File is being read.")
		val input = Source.fromFile(f)
		val lines = input.getLines.toArray
		val rev = lines.reverse 
		val writer = new PrintWriter(new File("rev.txt"))
		println("File is being written.")
		rev.foreach(writer.write)
		writer.close()
	}
}
