package fraud

import scala.math._

/**
 * Features
 * ========
 * Pure Scala feature computation over a List[StateRecord] window.
 *
 * This runs INSIDE flatMapGroupsWithState — no Spark window functions,
 * no DataFrames. Every method is a plain function over the in-memory
 * window list that is stored in AccountState.
 *
 * The math is identical to Features.py:
 *
 *   ComputeTimeFeatures  →  timeFeatures()
 *   ComputeAllFreqs      →  freqFeatures()
 *   ComputeAllEntropy    →  entropyFeatures()
 *
 * Window semantics
 * ----------------
 * window = last N StateRecords for this AccountNumber, oldest → newest.
 * "current" record is always window.last (already appended by the caller).
 * This is the exact equivalent of Spark's rowsBetween(-6, 0) where N=7.
 */
object Features {

  private val EPS = 1e-6
  private val TWO_PI = 2.0 * Pi

  // ── numeric helpers ────────────────────────────────────────────────────────

  private def safeLog(x: Double): Double = log(if (x > EPS) x else EPS)

  private def mean(xs: List[Double]): Double =
    if (xs.isEmpty) 0.0 else xs.sum / xs.size

  private def std(xs: List[Double]): Double = {
    if (xs.size < 2) return 0.0
    val mu = mean(xs)
    sqrt(xs.map(x => (x - mu) * (x - mu)).sum / xs.size)
  }

  // ── Time features ──────────────────────────────────────────────────────────
  //
  // Equivalent Spark window expressions:
  //   avg("LogAmount").over(roll)             →  mean(logAmounts)
  //   stddev("LogAmount").over(roll)          →  std(logAmounts)
  //   (LogAmount - MovingAvg)/(MovingStd + ε) →  zScore
  //   hour(Time), sin/cos encoding            →  hourSin / hourCos
  //   lag("Time",1).over(order)               →  window(n-2).epochSeconds
  //   log(TimeDiff)                           →  logTimeDiff

  def timeFeatures(window: List[StateRecord], current: StateRecord): (
    Double, Double, Double, Double, Double, Double, Double
  ) = {
    val logAmounts = window.map(_.logAmount)  // includes current (window.last)

    val logAmt  = current.logAmount
    val movAvg  = mean(logAmounts)
    val movStd  = std(logAmounts)
    val zScore  = (logAmt - movAvg) / (movStd + EPS)

    // Hour-of-day from epoch seconds (UTC)
    val hourOfDay = ((current.epochSeconds / 3600L) % 24L).toInt
    val hourSin   = sin(TWO_PI * hourOfDay / 24.0)
    val hourCos   = cos(TWO_PI * hourOfDay / 24.0)

    // Time diff to the previous record in the window
    val logTimeDiff: Double =
      if (window.size >= 2) {
        val diffSec   = current.epochSeconds - window(window.size - 2).epochSeconds
        val diffHours = diffSec.toDouble / 3600.0
        safeLog(diffHours)
      } else 0.0

    (logAmt, hourSin, hourCos, movAvg, movStd, zScore, logTimeDiff)
  }

  // ── Frequency features ─────────────────────────────────────────────────────
  //
  // Equivalent Spark expression (per categorical column):
  //   value_window = Window.partitionBy("AccountNumber", catCol).orderBy("Time").rowsBetween(-6,0)
  //   freq = count("*").over(value_window) / count("*").over(roll)
  //
  // Here: count how many records in the window share the current record's value,
  // divided by total window size.

  def freqFeatures(window: List[StateRecord], current: StateRecord): (
    Double, Double, Double, Double, Double, Double
  ) = {
    val n = window.size.toDouble
    if (n == 0) return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def freq(get: StateRecord => Option[String]): Double =
      window.count(r => get(r) == get(current)) / n

    (
      freq(_.transactionType),
      freq(_.channel),
      freq(_.cardType),
      freq(_.merchandGroup),
      freq(_.country),
      freq(_.city),
    )
  }

  // ── Entropy features ───────────────────────────────────────────────────────
  //
  // Equivalent Spark expression (per categorical column):
  //   n_unique   = spark_max(dense_rank().over(rankWin)).over(roll)
  //   p          = value_count / total_count
  //   rawEntropy = -sum((p * log(p)) / value_count).over(roll)
  //   entropy    = when(n_unique > 1, rawEntropy / log(n_unique)).otherwise(0)
  //
  // Correctly reduces to normalised Shannon entropy:
  //   H_norm = -Σ p_i * log(p_i) / log(n_unique)

  def entropyFeatures(window: List[StateRecord]): (
    Double, Double, Double, Double, Double, Double
  ) = {
    val n = window.size.toDouble
    if (n == 0) return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def normEntropy(get: StateRecord => Option[String]): Double = {
      val counts  = window.groupBy(get).map { case (k, vs) => k -> vs.size.toDouble }
      val nUnique = counts.size
      if (nUnique <= 1) return 0.0
      val h = counts.values.foldLeft(0.0) { (acc, c) =>
        val p = c / n
        acc - p * log(p)
      }
      h / log(nUnique.toDouble)
    }

    (
      normEntropy(_.transactionType),
      normEntropy(_.channel),
      normEntropy(_.cardType),
      normEntropy(_.merchandGroup),
      normEntropy(_.country),
      normEntropy(_.city),
    )
  }

  // ── Combined entry point ───────────────────────────────────────────────────

  def computeAll(window: List[StateRecord], current: StateRecord, age: Double): FeatureVector = {
    val (logAmt, hourSin, hourCos, movAvg, movStd, zScore, logTimeDiff) =
      timeFeatures(window, current)

    val (txnTypeFreq, chanFreq, cardFreq, mercFreq, cntryFreq, cityFreq) =
      freqFeatures(window, current)

    val (txnTypeEntr, chanEntr, cardEntr, mercEntr, cntryEntr, cityEntr) =
      entropyFeatures(window)

    FeatureVector(
      LogAmount    = logAmt,
      HourSin      = hourSin,
      HourCos      = hourCos,
      MovingAvg    = movAvg,
      MovingStd    = movStd,
      AmountZScore = zScore,
      LogTimeDiff  = logTimeDiff,
      Age          = age,
      TransactionTypeFreq    = txnTypeFreq,
      ChannelFreq            = chanFreq,
      CardTypeFreq           = cardFreq,
      MerchandFreq           = mercFreq,
      CountryFreq            = cntryFreq,
      CityFreq               = cityFreq,
      TransactionTypeEntropy = txnTypeEntr,
      ChannelEntropy         = chanEntr,
      CardTypeEntropy        = cardEntr,
      MerchandEntropy        = mercEntr,
      CountryEntropy         = cntryEntr,
      CityEntropy            = cityEntr,
    )
  }
}
