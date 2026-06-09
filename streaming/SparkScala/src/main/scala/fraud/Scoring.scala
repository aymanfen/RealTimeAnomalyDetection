package fraud

import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.scala.DefaultScalaModule
import org.slf4j.LoggerFactory

import java.io.{BufferedReader, DataOutputStream, InputStreamReader}
import java.net.{HttpURLConnection, URL}
import java.nio.charset.StandardCharsets
import scala.collection.JavaConverters._

/**
 * Scoring
 * =======
 * HTTP client for the model REST server.
 *
 * Uses only java.net.HttpURLConnection + Jackson — both already present on
 * every Spark executor.  No circe, no cats, no sttp → zero classpath conflicts.
 *
 * The ObjectMapper is @transient lazy so each executor JVM builds one instance
 * and reuses it across all records it processes.
 */
object Scoring {

  private val log = LoggerFactory.getLogger(getClass)

  val ModelServer: String = "http://localhost:8080"

  @transient private lazy val mapper: ObjectMapper = {
    val m = new ObjectMapper()
    m.registerModule(DefaultScalaModule)
    m
  }

  // ── Public API ─────────────────────────────────────────────────────────────

  def scoreSom(fv: FeatureVector): (Double, Map[String, Double]) = {
    try {
      val responseBody = post(
        s"$ModelServer/som/explain",
        mapper.writeValueAsString(Map("instances" -> List(fv.toList)))
      )

      // Parse response: { "scores": [0.123], "explanations": [{"feat": 0.01, ...}] }
      val root         = mapper.readTree(responseBody)
      val score        = root.get("scores").get(0).asDouble(0.0)
      val explainNode  = root.get("explanations").get(0)

      val explain: Map[String, Double] = asScalaIterator(explainNode.fields())
          .map(e => e.getKey -> e.getValue.asDouble(0.0))
          .toMap

      (score, explain)

    } catch {
      case ex: Exception =>
        log.error("Model server error: {}", ex.getMessage: AnyRef)
        (0.0, Map.empty[String, Double])
    }
  }

  // ── Raw HTTP POST — no external deps ───────────────────────────────────────

  private def post(urlStr: String, jsonBody: String, timeoutMs: Int = 60000): String = {
    val url  = new URL(urlStr)
    val conn = url.openConnection().asInstanceOf[HttpURLConnection]
    try {
      conn.setRequestMethod("POST")
      conn.setRequestProperty("Content-Type", "application/json; charset=UTF-8")
      conn.setConnectTimeout(timeoutMs)
      conn.setReadTimeout(timeoutMs)
      conn.setDoOutput(true)

      val bytes = jsonBody.getBytes(StandardCharsets.UTF_8)
      conn.setRequestProperty("Content-Length", bytes.length.toString)

      val out = new DataOutputStream(conn.getOutputStream)
      try { out.write(bytes) } finally { out.flush(); out.close() }

      val code = conn.getResponseCode
      val stream = if (code >= 200 && code < 300) conn.getInputStream
                   else conn.getErrorStream

      val reader = new BufferedReader(new InputStreamReader(stream, StandardCharsets.UTF_8))
      try {
        val sb = new StringBuilder
        var line = reader.readLine()
        while (line != null) { sb.append(line); line = reader.readLine() }
        if (code >= 400) throw new RuntimeException(s"HTTP $code: ${sb.toString.take(200)}")
        sb.toString
      } finally { reader.close() }

    } finally { conn.disconnect() }
  }
}
