"""
Collector: polls MMT every 60 seconds for VD, stats, and candles.
On startup, checks for gaps and backfills automatically.
"""
import time
import traceback
from config import EXCHANGE, SYMBOL
from mmt_client import MMTClient
from db import get_conn, insert_vd, insert_stats, insert_candles

POLL_INTERVAL = 60  # seconds
CHUNK_SECONDS = 6 * 3600  # 6-hour chunks for backfill


def get_last_timestamp(conn):
    """Get the most recent timestamp across all tables."""
    latest = 0
    for table in ["vd", "stats", "candles"]:
        row = conn.execute(
            f"SELECT MAX(timestamp) FROM {table} WHERE exchange=? AND symbol=?",
            (EXCHANGE, SYMBOL)
        ).fetchone()
        if row[0] and row[0] > latest:
            latest = row[0]
    return latest


def fill_gap(client, conn):
    """Check for gaps since last record and backfill if needed."""
    last_ts = get_last_timestamp(conn)
    now = int(time.time())

    if last_ts == 0:
        print("No existing data — run backfill.py first for historical data")
        return

    gap_seconds = now - last_ts
    gap_minutes = gap_seconds / 60

    if gap_minutes < 3:
        print(f"No gap (last record {gap_minutes:.0f}m ago)")
        return

    print(f"Gap detected: {gap_minutes:.0f} minutes ({gap_seconds/3600:.1f}h)")
    print(f"Backfilling from {time.strftime('%Y-%m-%d %H:%M', time.gmtime(last_ts))} to now...")

    chunk_start = last_ts
    chunks_done = 0

    while chunk_start < now:
        chunk_end = min(chunk_start + CHUNK_SECONDS, now)

        try:
            vd = client.get_vd(EXCHANGE, SYMBOL, tf="1m", from_ts=chunk_start, to_ts=chunk_end)
            if vd.get("data"):
                insert_vd(conn, EXCHANGE, SYMBOL, vd["data"])

            stats = client.get_stats(EXCHANGE, SYMBOL, tf="1m", from_ts=chunk_start, to_ts=chunk_end)
            if stats.get("data"):
                insert_stats(conn, EXCHANGE, SYMBOL, stats["data"])

            candles = client.get_candles(EXCHANGE, SYMBOL, tf="1m", from_ts=chunk_start, to_ts=chunk_end)
            if candles.get("data"):
                insert_candles(conn, EXCHANGE, SYMBOL, candles["data"])

            chunks_done += 1
            vd_n = len(vd.get("data", []))
            stats_n = len(stats.get("data", []))
            print(f"  Chunk {chunks_done}: {time.strftime('%m/%d %H:%M', time.gmtime(chunk_start))} → "
                  f"{vd_n} VD + {stats_n} stats")

        except Exception as e:
            print(f"  Error at {chunk_start}: {e}")
            time.sleep(5)

        time.sleep(2)  # rate limit
        chunk_start = chunk_end

    print(f"Gap fill complete: {chunks_done} chunks")


def collect_once(client, conn):
    now = int(time.time())
    from_ts = now - 120

    vd_resp = client.get_vd(EXCHANGE, SYMBOL, tf="1m", from_ts=from_ts, to_ts=now)
    if vd_resp.get("data"):
        insert_vd(conn, EXCHANGE, SYMBOL, vd_resp["data"])

    stats_resp = client.get_stats(EXCHANGE, SYMBOL, tf="1m", from_ts=from_ts, to_ts=now)
    if stats_resp.get("data"):
        insert_stats(conn, EXCHANGE, SYMBOL, stats_resp["data"])

    candles_resp = client.get_candles(EXCHANGE, SYMBOL, tf="1m", from_ts=from_ts, to_ts=now)
    if candles_resp.get("data"):
        insert_candles(conn, EXCHANGE, SYMBOL, candles_resp["data"])

    return len(vd_resp.get("data", [])), len(stats_resp.get("data", []))


def main():
    client = MMTClient()
    conn = get_conn()
    print(f"Collector started: {EXCHANGE} {SYMBOL}")

    # Fill any gaps first
    fill_gap(client, conn)

    print(f"Polling every {POLL_INTERVAL}s")
    while True:
        try:
            vd_n, stats_n = collect_once(client, conn)
            print(f"[{time.strftime('%H:%M:%S')}] {vd_n} VD + {stats_n} stats")
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] Error: {e}")
            traceback.print_exc()
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
