"""
Collector: polls MMT every 60 seconds for VD, stats, and candles.
Covers all spot exchanges. On startup, checks for gaps and backfills automatically.
"""
import time
import traceback
from config import EXCHANGES, SYMBOL
from mmt_client import MMTClient
from db import get_conn, insert_vd, insert_stats, insert_candles

POLL_INTERVAL = 60  # seconds
CHUNK_SECONDS = 6 * 3600  # 6-hour chunks for backfill


def get_last_timestamp(conn, exchange):
    """Get the most recent timestamp across all tables for an exchange."""
    latest = 0
    for table in ["vd", "stats", "candles"]:
        row = conn.execute(
            f"SELECT MAX(timestamp) FROM {table} WHERE exchange=? AND symbol=?",
            (exchange, SYMBOL)
        ).fetchone()
        if row[0] and row[0] > latest:
            latest = row[0]
    return latest


def fill_gap(client, conn, exchange):
    """Check for gaps since last record and backfill if needed."""
    last_ts = get_last_timestamp(conn, exchange)
    now = int(time.time())

    if last_ts == 0:
        print(f"  [{exchange}] No existing data — run backfill.py first")
        return

    gap_seconds = now - last_ts
    gap_minutes = gap_seconds / 60

    if gap_minutes < 3:
        print(f"  [{exchange}] No gap ({gap_minutes:.0f}m ago)")
        return

    print(f"  [{exchange}] Gap: {gap_minutes:.0f}m ({gap_seconds/3600:.1f}h) — backfilling...")

    chunk_start = last_ts
    chunks_done = 0

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
            vd_n = len(vd.get("data", []))
            stats_n = len(stats.get("data", []))
            print(f"    Chunk {chunks_done}: {time.strftime('%m/%d %H:%M', time.gmtime(chunk_start))} → "
                  f"{vd_n} VD + {stats_n} stats")

        except Exception as e:
            print(f"    Error at {chunk_start}: {e}")
            time.sleep(5)

        time.sleep(2)  # rate limit
        chunk_start = chunk_end

    print(f"  [{exchange}] Gap fill complete: {chunks_done} chunks")


def collect_once(client, conn, exchange):
    now = int(time.time())
    from_ts = now - 120

    vd_resp = client.get_vd(exchange, SYMBOL, tf="1m", from_ts=from_ts, to_ts=now)
    if vd_resp.get("data"):
        insert_vd(conn, exchange, SYMBOL, vd_resp["data"])

    stats_resp = client.get_stats(exchange, SYMBOL, tf="1m", from_ts=from_ts, to_ts=now)
    if stats_resp.get("data"):
        insert_stats(conn, exchange, SYMBOL, stats_resp["data"])

    candles_resp = client.get_candles(exchange, SYMBOL, tf="1m", from_ts=from_ts, to_ts=now)
    if candles_resp.get("data"):
        insert_candles(conn, exchange, SYMBOL, candles_resp["data"])

    return len(vd_resp.get("data", [])), len(stats_resp.get("data", []))


def main():
    client = MMTClient()
    conn = get_conn()
    exchanges = list(EXCHANGES.keys())

    print(f"Collector started: {len(exchanges)} exchanges × {SYMBOL}")
    print(f"Exchanges: {', '.join(exchanges)}")
    print(f"Rate budget: {len(exchanges) * 3} weight/min (limit: 100)")

    # Fill gaps for all exchanges
    for exch in exchanges:
        fill_gap(client, conn, exch)

    print(f"\nPolling every {POLL_INTERVAL}s")
    while True:
        try:
            parts = []
            for exch in exchanges:
                vd_n, stats_n = collect_once(client, conn, exch)
                parts.append(f"{exch}:{vd_n}vd")
            print(f"[{time.strftime('%H:%M:%S')}] {' | '.join(parts)}")
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] Error: {e}")
            traceback.print_exc()
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
