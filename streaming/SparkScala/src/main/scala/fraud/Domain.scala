package fraud

import java.sql.Timestamp
import org.apache.spark.sql.types._
import org.apache.spark.sql.Encoder
import org.apache.spark.sql.Encoders

// ── Raw transaction arriving from Kafka JSON ───────────────────────────────────
case class Transaction(
  TransactionID:     String,
  Time:              Option[Timestamp],
  AccountNumber:     String,
  CardNumber:        Option[String],
  TransactionType:   Option[String],
  Channel:           Option[String],
  TransactionAmount: Double,
  MerchandCode:      Option[String],
  MerchandGroup:     Option[String],
  Country:           Option[String],
  City:              Option[String],
  Country2:          Option[String],
  City2:             Option[String],
  CardType:          Option[String],
  Age:               Option[Int],
  Gender:            Option[String],
  Bank:              Option[String],
)

// ── Compact record stored in state (only what feature computation needs) ────────
//
// We do NOT store the full Transaction in state to keep the state footprint small.
// Time is stored as epoch-seconds (Long) — Timestamp is not Kryo-friendly in state.
case class StateRecord(
  epochSeconds:    Long,
  logAmount:       Double,
  transactionType: Option[String],
  channel:         Option[String],
  cardType:        Option[String],
  merchandGroup:   Option[String],
  country:         Option[String],
  city:            Option[String],
)

// ── State stored per AccountNumber key ─────────────────────────────────────────
//
// window   : last N StateRecords, ordered oldest → newest
// This is what survives across micro-batches and is updated on every new record.
case class AccountState(
  window: List[StateRecord],  // oldest → newest, max length = windowSize
)

// ── Computed feature vector ─────────────────────────────────────────────────────
case class FeatureVector(
  LogAmount:    Double,
  HourSin:      Double,
  HourCos:      Double,
  MovingAvg:    Double,
  MovingStd:    Double,
  AmountZScore: Double,
  LogTimeDiff:  Double,
  Age:          Double,
  TransactionTypeFreq:    Double,
  ChannelFreq:            Double,
  CardTypeFreq:           Double,
  MerchandFreq:           Double,
  CountryFreq:            Double,
  CityFreq:               Double,
  TransactionTypeEntropy: Double,
  ChannelEntropy:         Double,
  CardTypeEntropy:        Double,
  MerchandEntropy:        Double,
  CountryEntropy:         Double,
  CityEntropy:            Double,
) {
  // Ordered to match FEATURE_COLS in the Python model server
  def toList: List[Double] = List(
    Age, LogAmount, AmountZScore, MovingAvg, MovingStd,
    LogTimeDiff, HourSin, HourCos,
    TransactionTypeFreq, ChannelFreq, CardTypeFreq, MerchandFreq, CountryFreq, CityFreq,
    TransactionTypeEntropy, ChannelEntropy, CardTypeEntropy, MerchandEntropy, CountryEntropy, CityEntropy,
  )
}

// ── Final enriched output record ────────────────────────────────────────────────
case class ScoredTransaction(
  TransactionID:     String,
  Time:              Option[Timestamp],
  AccountNumber:     String,
  CardNumber:        Option[String],
  TransactionType:   Option[String],
  Channel:           Option[String],
  TransactionAmount: Double,
  MerchandCode:      Option[String],
  MerchandGroup:     Option[String],
  Country:           Option[String],
  City:              Option[String],
  Country2:          Option[String],
  City2:             Option[String],
  CardType:          Option[String],
  Age:               Option[Int],
  Gender:            Option[String],
  Bank:              Option[String],
  // features
  LogAmount:    Double,
  HourSin:      Double,
  HourCos:      Double,
  MovingAvg:    Double,
  MovingStd:    Double,
  AmountZScore: Double,
  LogTimeDiff:  Double,
  TransactionTypeFreq:    Double,
  ChannelFreq:            Double,
  CardTypeFreq:           Double,
  MerchandFreq:           Double,
  CountryFreq:            Double,
  CityFreq:               Double,
  TransactionTypeEntropy: Double,
  ChannelEntropy:         Double,
  CardTypeEntropy:        Double,
  MerchandEntropy:        Double,
  CountryEntropy:         Double,
  CityEntropy:            Double,
  // model output
  modelScore:   Double,
  modelExplain: Map[String, Double],
  IsAnomaly:    Boolean,
  IsCritical:   Boolean,
)

// ── Schema for Kafka JSON parsing ───────────────────────────────────────────────
object Domain {

  val TransactionSchema: StructType = StructType(Seq(
    StructField("TransactionID",     StringType),
    StructField("Time",              TimestampType),
    StructField("AccountNumber",     StringType),
    StructField("CardNumber",        StringType),
    StructField("TransactionType",   StringType),
    StructField("Channel",           StringType),
    StructField("TransactionAmount", DoubleType),
    StructField("MerchandCode",      StringType),
    StructField("MerchandGroup",     StringType),
    StructField("Country",           StringType),
    StructField("City",              StringType),
    StructField("Country2",          StringType),
    StructField("City2",             StringType),
    StructField("CardType",          StringType),
    StructField("Age",               IntegerType),
    StructField("Gender",            StringType),
    StructField("Bank",              StringType),
  ))

  // Explicit encoders for the state and output types used in flatMapGroupsWithState
  implicit val transactionEncoder: Encoder[Transaction]         = Encoders.product
  implicit val accountStateEncoder: Encoder[AccountState]       = Encoders.product
  implicit val scoredTxnEncoder: Encoder[ScoredTransaction]     = Encoders.product
}
