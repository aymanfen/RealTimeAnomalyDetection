package fraud

import fraud.Domain._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.streaming.{GroupStateTimeout, OutputMode, StreamingQuery}
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}

/**
 * FraudPipeline
 * =============
 * Main entry point.
 *
 * The critical difference from the previous (non-stateful) Scala version:
 *
 *   OLD:  df.writeStream.foreachBatch { batchDf =>
 *             batchDf.transform(Features.computeTimeFeatures)  // window functions
 *         }                                                      // ← blind to previous batches
 *
 *   NEW:  df.as[Transaction]
 *           .groupByKey(_.AccountNumber)
 *           .flatMapGroupsWithState(                            // ← true stateful API
 *               OutputMode.Append,
 *               GroupStateTimeout.NoTimeout
 *             )(StatefulScoringFunction.apply)                  // ← window lives in GroupState
 *
 * Iceberg writes
 * --------------
 * Because flatMapGroupsWithState produces a streaming Dataset[ScoredTransaction],
 * we cannot call writeTo().append() directly (Iceberg streaming write requires
 * format("iceberg")). We split the stream:
 *   - foreachBatch on the scored stream handles Bronze + Gold Iceberg writes
 *   - the stateful operator itself handles Silver (features + scores)
 *
 * Run:
 *   spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1,org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.4.2 --class fraud.FraudPipeline target/scala-2.12/spark-fraud-stateful-assembly-1.0.0.jar
 */
object FraudPipeline {

  // ── Config ─────────────────────────────────────────────────────────────────
  val KafkaBootstrap:  String = "192.168.1.14:9092"
  val KafkaTopic:      String = "transactions"
  val CheckpointBase:  String = "hdfs://192.168.1.14:9000/checkpoints/lakehouse-stateful"
  val Warehouse:       String = "hdfs://192.168.1.14:9000/iceberg"

  // ── Spark session ──────────────────────────────────────────────────────────
  val spark: SparkSession = SparkSession.builder()
    .appName("SparkFraudStateful")
    .config("spark.sql.extensions",
      "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
    .config("spark.sql.catalog.lake",            "org.apache.iceberg.spark.SparkCatalog")
    .config("spark.sql.catalog.lake.type",       "hadoop")
    .config("spark.sql.catalog.lake.warehouse",  Warehouse)
    .config("spark.sql.shuffle.partitions",      "4")
    // Required for flatMapGroupsWithState with streaming Dataset
    .config("spark.sql.streaming.statefulOperator.checkCorrectness.enabled", "false")
    .getOrCreate()

  spark.sparkContext.setLogLevel("WARN")

  import spark.implicits._

  // ── foreachBatch sink — Bronze + Gold Iceberg ─────────────────────────────

  def lakehouseWrite(batchDs: Dataset[ScoredTransaction], batchId: Long): Unit = {
    val batchDf = batchDs.toDF()

    // Bronze — raw fields only, no features
    batchDf
      .select(
        "TransactionID", "Time", "AccountNumber", "CardNumber",
        "TransactionType", "Channel", "TransactionAmount",
        "MerchandCode", "MerchandGroup", "Country", "City",
        "Country2", "City2", "CardType", "Age", "Gender", "Bank",
      )
      .withColumn("bronzetimestamp", current_timestamp())
      .writeTo("lake.bronze.transactions").append()

    // Gold — transaction fact (scores + anomaly flags)
    batchDf
      .select(
        "TransactionID", "Time", "AccountNumber", "MerchandCode",
        "TransactionType", "Channel", "TransactionAmount",
        "Country", "City",
        "modelScore", "IsAnomaly", "IsCritical",
      )
      .withColumnRenamed("modelScore", "modelscore")
      .withColumn("goldtimestamp", current_timestamp())
      .writeTo("lake.gold.TransactionFact").append()

    // Gold — explain fact (explode the Map[String,Double] → one row per feature)
    batchDf
      .select($"TransactionID", $"Time", explode($"modelExplain").as(Seq("FeatureName", "FeatureValue")))
      .select("TransactionID", "Time", "FeatureName", "FeatureValue")
      .writeTo("lake.gold.ExplainFact").append()

    // Silver — full feature vector + scores
    batchDf
      .select("TransactionID", "Time", "AccountNumber", "CardNumber",
        "TransactionType", "Channel", "TransactionAmount",
        "MerchandCode", "MerchandGroup", "Country", "City",
        "Country2", "City2", "CardType", "Age", "Gender", "Bank",
        "LogAmount","HourCos","HourSin","MovingAvg","MovingStd","AmountZScore","LogTimeDiff",
        "TransactionTypeFreq","ChannelFreq","CardTypeFreq","MerchandFreq","CountryFreq","CityFreq",
        "TransactionTypeEntropy","ChannelEntropy","CardTypeEntropy","MerchandEntropy","CountryEntropy","CityEntropy",
        "modelScore","modelExplain")
      .withColumnRenamed("modelScore",   "modelscore")
      .withColumnRenamed("modelExplain", "modelexplain")
      .withColumn("silvertimestamp", current_timestamp())
      .writeTo("lake.silver.features").append()
  }

  // ── Main ───────────────────────────────────────────────────────────────────

  def main(args: Array[String]): Unit = {

    // ── 1. Read from Kafka ────────────────────────────────────────────────
    val kafkaDf = spark.readStream
      .format("kafka")
      .option("kafka.bootstrap.servers", KafkaBootstrap)
      .option("subscribe",               KafkaTopic)
      .option("startingOffsets",         "latest")
      .option("failOnDataLoss",          "false")
      .load()

    // ── 2. Parse JSON → typed Dataset[Transaction] ────────────────────────
    val transactions: Dataset[Transaction] = kafkaDf
      .selectExpr("CAST(value AS STRING)")
      .select(from_json(col("value"), Domain.TransactionSchema).alias("data"))
      .select("data.*")
      .filter(col("AccountNumber").isNotNull)
      .as[Transaction]

    // ── 3. flatMapGroupsWithState — the stateful core ─────────────────────
    //
    // groupByKey        : partition by AccountNumber (same as Window.partitionBy)
    // flatMapGroupsWithState : for each (key, micro-batch records, state) → outputs
    //
    // GroupStateTimeout.NoTimeout  : state lives forever (until you remove it).
    //   Switch to ProcessingTimeTimeout and call state.remove() in the timeout
    //   branch of StatefulScoringFunction if you want to evict idle accounts.
    //
    // OutputMode.Append : each call emits new rows; no updates to existing rows.

    val scored: Dataset[ScoredTransaction] = transactions
      .groupByKey(_.AccountNumber)
      .flatMapGroupsWithState[AccountState, ScoredTransaction](
        outputMode = OutputMode.Append(),
        timeoutConf = GroupStateTimeout.NoTimeout,
      )(StatefulScoringFunction.apply)

    // ── 4. Write to Iceberg via foreachBatch ──────────────────────────────
    val query: StreamingQuery = scored
      .writeStream
      .foreachBatch(lakehouseWrite _)
      .option("checkpointLocation", s"$CheckpointBase/scored")
      .outputMode("append")
      .start()

    query.awaitTermination()
  }
}
