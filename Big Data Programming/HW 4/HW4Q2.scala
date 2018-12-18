import org.apache.spark.streaming.{Seconds, StreamingContext}
import org.apache.spark.streaming.twitter._
import org.apache.spark.SparkConf
import org.apache.log4j.{Level, Logger}


object HW4Q2 {
  
  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.ERROR)

    var consumerKey = "PttEZFyQIhrxfqmIBfZ9x5fKc"
    var consumerSecret = "RyCgiLKqP8kQjcwxJ8h9EQFzLGm3dL5n2eCTN9YpQ2RRYG3cd7"
    var accessToken = "931749427211128832-UJP8jUVAEieK0fP9mmHn5yuD4DiGi8M"
    var accessTokenSecret = "TAuHxwpL6FghEom8IDUtQTQPUeHik6nxhjmhzvyGlfGUk"

  if (args.length > 3) {
    val  Array(consumerKey, consumerSecret, accessToken, accessTokenSecret) = args.take(4)
    }

    System.setProperty("twitter4j.oauth.consumerKey", consumerKey)
    System.setProperty("twitter4j.oauth.consumerSecret", consumerSecret)
    System.setProperty("twitter4j.oauth.accessToken", accessToken)
    System.setProperty("twitter4j.oauth.accessTokenSecret", accessTokenSecret)

    val sparkConf = new SparkConf().setAppName("Twitter").setMaster("local[2]")
    val ssc = new StreamingContext(sparkConf, Seconds(3))
    val stream = TwitterUtils.createStream(ssc, None)

    val statuses = stream.map(status => status.getText())
    statuses.print()

    ssc.start()
    ssc.awaitTermination()

  }

}

