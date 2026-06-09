package fraud

import org.apache.spark.sql.streaming.{GroupState, GroupStateTimeout, OutputMode}
import org.slf4j.LoggerFactory

import java.sql.Timestamp

/**
 * StatefulScoringFunction
 * =======================
 * The function passed to flatMapGroupsWithState.
 *
 * Signature required by Spark:
 *   (key: K, inputs: Iterator[V], state: GroupState[S]) => Iterator[U]
 *
 *   K = String          (AccountNumber)
 *   V = Transaction     (one row per record in the micro-batch)
 *   S = AccountState    (persistent state: the sliding window of last N records)
 *   U = ScoredTransaction (output record emitted downstream)
 *
 * What happens on each call
 * -------------------------
 * Spark calls this function once per (AccountNumber, micro-batch).
 * `inputs` contains all transactions for that account in this batch —
 * we process them ONE BY ONE in time order so the window grows correctly.
 *
 * For each transaction:
 *   1. Convert to StateRecord (compact form stored in state)
 *   2. Append to the window, trim to windowSize                  ← cross-batch state update
 *   3. Compute features over the full in-memory window           ← Features.computeAll
 *   4. Call model REST server                                     ← Scoring.scoreSom
 *   5. Yield ScoredTransaction                                    ← goes downstream
 *
 * State persistence
 * -----------------
 * AccountState is checkpointed by Spark between micro-batches.
 * On the next batch for the same account, window already contains the
 * records from all previous batches — giving TRUE cross-batch sliding windows.
 *
 * Timeout
 * -------
 * We use NoTimeout. Accounts that stop sending transactions simply keep
 * their last window in state indefinitely. Switch to ProcessingTimeTimeout
 * and call state.remove() in the timeout branch if you want to evict stale
 * accounts and reclaim state memory.
 */
object StatefulScoringFunction {

  private val log = LoggerFactory.getLogger(getClass)

  val WindowSize:        Int    = 7     // last N transactions — matches rowsBetween(-6,0)
  val AnomalyThreshold:  Double = 3.5
  val CriticalThreshold: Double = 5.25

  // ── Entry point — pass this directly to flatMapGroupsWithState ────────────

  def apply(
    accountNumber: String,
    inputs:        Iterator[Transaction],
    state:         GroupState[AccountState],
  ): Iterator[ScoredTransaction] = {

    // Restore existing window from state, or start empty
    val existingWindow: List[StateRecord] =
      if (state.exists) state.get.window else List.empty

    // Sort inputs by time within this micro-batch to preserve ordering.
    // (Kafka partitions guarantee ordering per-partition, but multiple
    //  partitions for one account can arrive out-of-order in the same batch.)
    val sortedInputs = inputs.toList.sortBy(_.Time.map(_.getTime).getOrElse(0L))

    // Process each transaction, threading the window forward
    var currentWindow = existingWindow

    val outputs: List[ScoredTransaction] = sortedInputs.flatMap { txn =>

      // ── 1. Build compact StateRecord ──────────────────────────────────────
      val epochSec = txn.Time.map(_.getTime / 1000L).getOrElse(0L)
      val logAmt   = math.log(math.max(txn.TransactionAmount, 1e-6))

      val sr = StateRecord(
        epochSeconds    = epochSec,
        logAmount       = logAmt,
        transactionType = txn.TransactionType,
        channel         = txn.Channel,
        cardType        = txn.CardType,
        merchandGroup   = txn.MerchandGroup,
        country         = txn.Country,
        city            = txn.City,
      )

      // ── 2. Slide the window ───────────────────────────────────────────────
      //  append current, trim to WindowSize (oldest record falls off)
      currentWindow = (currentWindow :+ sr).takeRight(WindowSize)

      // ── 3. Compute features ───────────────────────────────────────────────
      val fv = Features.computeAll(
        window  = currentWindow,
        current = sr,
        age     = txn.Age.map(_.toDouble).getOrElse(0.0),
      )

      // ── 4. Score ──────────────────────────────────────────────────────────
      val (score, explain) = Scoring.scoreSom(fv)


      // ── 5. Emit ───────────────────────────────────────────────────────────
      Some(ScoredTransaction(
        TransactionID     = txn.TransactionID,
        Time              = txn.Time,
        AccountNumber     = txn.AccountNumber,
        CardNumber        = txn.CardNumber,
        TransactionType   = txn.TransactionType,
        Channel           = txn.Channel,
        TransactionAmount = txn.TransactionAmount,
        MerchandCode      = txn.MerchandCode,
        MerchandGroup     = txn.MerchandGroup,
        Country           = txn.Country,
        City              = txn.City,
        Country2          = txn.Country2,
        City2             = txn.City2,
        CardType          = txn.CardType,
        Age               = txn.Age,
        Gender            = txn.Gender,
        Bank              = txn.Bank,
        LogAmount         = fv.LogAmount,
        HourSin           = fv.HourSin,
        HourCos           = fv.HourCos,
        MovingAvg         = fv.MovingAvg,
        MovingStd         = fv.MovingStd,
        AmountZScore      = fv.AmountZScore,
        LogTimeDiff       = fv.LogTimeDiff,
        TransactionTypeFreq    = fv.TransactionTypeFreq,
        ChannelFreq            = fv.ChannelFreq,
        CardTypeFreq           = fv.CardTypeFreq,
        MerchandFreq           = fv.MerchandFreq,
        CountryFreq            = fv.CountryFreq,
        CityFreq               = fv.CityFreq,
        TransactionTypeEntropy = fv.TransactionTypeEntropy,
        ChannelEntropy         = fv.ChannelEntropy,
        CardTypeEntropy        = fv.CardTypeEntropy,
        MerchandEntropy        = fv.MerchandEntropy,
        CountryEntropy         = fv.CountryEntropy,
        CityEntropy            = fv.CityEntropy,
        modelScore             = score,
        modelExplain           = explain,
        IsAnomaly              = score > AnomalyThreshold,
        IsCritical             = score > CriticalThreshold,
      ))
    }

    // ── Persist updated window back into Spark-managed state ─────────────────
    // This is what makes it truly stateful — Spark checkpoints this and
    // restores it at the start of the next micro-batch for this account.
    state.update(AccountState(window = currentWindow))

    outputs.iterator
  }
}
