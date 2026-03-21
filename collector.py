"""
Collector: polls MMT every 60 seconds for VD, stats, and candles.
Stores 1-minute snapshots to SQLite.
"""
import time
import traceback
from config import EXCHANGE, SYMBOL
from mmt_client import MMTClient
from db import get_conn, insert_vd, insert_stats, insert_candles

POLL_INTERVAL = 60  # seconds


def collect_once(client, conn):
    now = int(time.time())
    from_ts = now - 120  # last 2 minutes to catch any gaps

    # VD
    vd_resp = client.get_vd(EXCHANGE, SYMBOL, tf="1m", from_ts=from_ts, to_ts=now)
    if vd_resp.get("data"):
        insert_vd(conn, EXCHANGE, SYMBOL, vd_resp["data"])

    # Stats
    stats_resp = client.get_stats(EXCHANGE, SYMBOL, tf="1m", from_ts=from_ts, to_ts=now)
    if stats_resp.get("data"):
        insert_stats(conn, EXCHANGE, SYMBOL, stats_resp["data"])

    # Candles
    candles_resp = client.get_candles(EXCHANGE, SYMBOL, tf="1m", from_ts=from_ts, to_ts=now)
    if candles_resp.get("data"):
        insert_candles(conn, EXCHANGE, SYMBOL, candles_resp["data"])

    return len(vd_resp.get("data", [])), len(stats_resp.get("data", []))


def main():
    client = MMTClient()
    conn = get_conn()
    print(f"Collector started: {EXCHANGE} {SYMBOL}")
    print(f"Polling every {POLL_INTERVAL}s")

    while True:
        try:
            vd_n, stats_n = collect_once(client, conn)
            print(f"[{time.strftime('%H:%M:%S')}] Collected {vd_n} VD + {stats_n} stats records")
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] Error: {e}")
            traceback.print_exc()
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
