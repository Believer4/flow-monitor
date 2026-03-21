"""
Backfill VD, stats, and candles in 6-hour chunks going back HISTORY_DAYS.
3 endpoints × 1 weight each = 3 weight per chunk.
~360 chunks for 90 days = ~1080 weight total, spread across ~11 minutes at 100/min.
"""
import time
from config import EXCHANGE, SYMBOL, HISTORY_DAYS
from mmt_client import MMTClient
from db import get_conn, insert_vd, insert_stats, insert_candles, get_record_count

CHUNK_SECONDS = 6 * 3600  # 6 hours


def backfill():
    client = MMTClient()
    conn = get_conn()

    now = int(time.time())
    start = now - (HISTORY_DAYS * 86400)

    print(f"Backfilling {HISTORY_DAYS} days: {EXCHANGE} {SYMBOL}")
    print(f"From {time.strftime('%Y-%m-%d', time.gmtime(start))} to {time.strftime('%Y-%m-%d', time.gmtime(now))}")

    before_vd = get_record_count(conn, "vd", EXCHANGE, SYMBOL)
    before_stats = get_record_count(conn, "stats", EXCHANGE, SYMBOL)
    before_candles = get_record_count(conn, "candles", EXCHANGE, SYMBOL)

    chunk_start = start
    chunks_done = 0
    total_chunks = (now - start) // CHUNK_SECONDS + 1

    while chunk_start < now:
        chunk_end = min(chunk_start + CHUNK_SECONDS, now)

        try:
            # VD
            vd = client.get_vd(EXCHANGE, SYMBOL, tf="1m", from_ts=chunk_start, to_ts=chunk_end)
            if vd.get("data"):
                insert_vd(conn, EXCHANGE, SYMBOL, vd["data"])

            # Stats
            stats = client.get_stats(EXCHANGE, SYMBOL, tf="1m", from_ts=chunk_start, to_ts=chunk_end)
            if stats.get("data"):
                insert_stats(conn, EXCHANGE, SYMBOL, stats["data"])

            # Candles
            candles = client.get_candles(EXCHANGE, SYMBOL, tf="1m", from_ts=chunk_start, to_ts=chunk_end)
            if candles.get("data"):
                insert_candles(conn, EXCHANGE, SYMBOL, candles["data"])

            chunks_done += 1
            if chunks_done % 10 == 0:
                print(f"  [{chunks_done}/{total_chunks}] {time.strftime('%Y-%m-%d %H:%M', time.gmtime(chunk_start))}")

        except Exception as e:
            print(f"  Error at {chunk_start}: {e}")
            time.sleep(5)
            chunk_start = chunk_end
            continue

        # Rate limit: 3 calls per chunk, 100 weight/min budget
        # ~33 chunks/min is safe, sleep 2s between chunks
        time.sleep(2)
        chunk_start = chunk_end

    after_vd = get_record_count(conn, "vd", EXCHANGE, SYMBOL)
    after_stats = get_record_count(conn, "stats", EXCHANGE, SYMBOL)
    after_candles = get_record_count(conn, "candles", EXCHANGE, SYMBOL)

    print(f"\nBackfill complete:")
    print(f"  VD:      {before_vd} → {after_vd} (+{after_vd - before_vd})")
    print(f"  Stats:   {before_stats} → {after_stats} (+{after_stats - before_stats})")
    print(f"  Candles: {before_candles} → {after_candles} (+{after_candles - before_candles})")


if __name__ == "__main__":
    backfill()
