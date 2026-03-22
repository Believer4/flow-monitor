"""
Backfill VD, stats, and candles in 6-hour chunks going back HISTORY_DAYS.
Runs all spot exchanges sequentially.

Per exchange: 3 endpoints × 1 weight each = 3 weight per chunk.
~360 chunks for 90 days × 4 exchanges = ~1440 chunks total.
At 2s spacing = ~48 minutes.
"""
import sys
import time
from config import EXCHANGES, SYMBOL, HISTORY_DAYS
from mmt_client import MMTClient
from db import get_conn, insert_vd, insert_stats, insert_candles, get_record_count

CHUNK_SECONDS = 6 * 3600  # 6 hours


def backfill_exchange(client, conn, exchange):
    now = int(time.time())
    start = now - (HISTORY_DAYS * 86400)

    before_vd = get_record_count(conn, "vd", exchange, SYMBOL)
    before_stats = get_record_count(conn, "stats", exchange, SYMBOL)
    before_candles = get_record_count(conn, "candles", exchange, SYMBOL)

    # Skip if already backfilled
    if before_vd > 100000:
        print(f"  [{exchange}] Already has {before_vd:,} VD records — skipping")
        return

    print(f"  [{exchange}] Backfilling {HISTORY_DAYS} days...")

    chunk_start = start
    chunks_done = 0
    total_chunks = (now - start) // CHUNK_SECONDS + 1

    while chunk_start < now:
        chunk_end = min(chunk_start + CHUNK_SECONDS, now)

        try:
            vd = client.get_vd(exchange, SYMBOL, tf="1m", from_ts=chunk_start, to_ts=chunk_end)
            if vd.get("data"):
                insert_vd(conn, exchange, SYMBOL, vd["data"])

            stats = client.get_stats(exchange, SYMBOL, tf="1m", from_ts=chunk_start, to_ts=chunk_end)
            if stats.get("data"):
                insert_stats(conn, exchange, SYMBOL, stats["data"])

            candles = client.get_candles(exchange, SYMBOL, tf="1m", from_ts=chunk_start, to_ts=chunk_end)
            if candles.get("data"):
                insert_candles(conn, exchange, SYMBOL, candles["data"])

            chunks_done += 1
            if chunks_done % 20 == 0:
                pct = chunks_done / total_chunks * 100
                print(f"    [{chunks_done}/{total_chunks}] {pct:.0f}% — {time.strftime('%Y-%m-%d %H:%M', time.gmtime(chunk_start))}")

        except Exception as e:
            print(f"    Error at {chunk_start}: {e}")
            time.sleep(5)
            chunk_start = chunk_end
            continue

        time.sleep(2)
        chunk_start = chunk_end

    after_vd = get_record_count(conn, "vd", exchange, SYMBOL)
    after_stats = get_record_count(conn, "stats", exchange, SYMBOL)
    after_candles = get_record_count(conn, "candles", exchange, SYMBOL)

    print(f"  [{exchange}] Done: VD +{after_vd - before_vd:,} | Stats +{after_stats - before_stats:,} | Candles +{after_candles - before_candles:,}")


def backfill():
    client = MMTClient()
    conn = get_conn()

    # Allow specifying a single exchange: python3 backfill.py bybit
    if len(sys.argv) > 1:
        targets = [e for e in sys.argv[1:] if e in EXCHANGES]
        if not targets:
            print(f"Unknown exchange(s). Available: {', '.join(EXCHANGES.keys())}")
            return
    else:
        targets = list(EXCHANGES.keys())

    print(f"Backfilling {len(targets)} exchange(s): {', '.join(targets)}")
    print(f"History: {HISTORY_DAYS} days | Chunk: {CHUNK_SECONDS//3600}h")

    for exch in targets:
        backfill_exchange(client, conn, exch)

    print("\nAll done.")


if __name__ == "__main__":
    backfill()
