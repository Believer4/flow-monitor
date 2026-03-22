import sqlite3
import os
import json

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "flow.db")


def get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS vd (
            timestamp INTEGER NOT NULL,
            exchange TEXT NOT NULL,
            symbol TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            n INTEGER,
            PRIMARY KEY (timestamp, exchange, symbol)
        );

        CREATE TABLE IF NOT EXISTS stats (
            timestamp INTEGER NOT NULL,
            exchange TEXT NOT NULL,
            symbol TEXT NOT NULL,
            last_price REAL,
            bid_depth TEXT,
            ask_depth TEXT,
            skew TEXT,
            buy_vol REAL,
            sell_vol REAL,
            PRIMARY KEY (timestamp, exchange, symbol)
        );

        CREATE TABLE IF NOT EXISTS candles (
            timestamp INTEGER NOT NULL,
            exchange TEXT NOT NULL,
            symbol TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            buy_vol REAL,
            sell_vol REAL,
            PRIMARY KEY (timestamp, exchange, symbol)
        );
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            exchange TEXT NOT NULL,
            symbol TEXT NOT NULL,
            price REAL,
            size_usd REAL,
            side TEXT,
            is_large INTEGER DEFAULT 0,
            z_score REAL DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_trades_ts ON trades(exchange, symbol, timestamp);

        CREATE TABLE IF NOT EXISTS custom_depth (
            timestamp INTEGER NOT NULL,
            exchange TEXT NOT NULL,
            symbol TEXT NOT NULL,
            last_price REAL,
            bid_depth_25 REAL,
            ask_depth_25 REAL,
            PRIMARY KEY (timestamp, exchange, symbol)
        );

        CREATE TABLE IF NOT EXISTS depth_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            exchange TEXT NOT NULL,
            symbol TEXT NOT NULL,
            price REAL,
            side TEXT,
            event_type TEXT,
            size_before REAL,
            size_after REAL,
            size_usd REAL,
            filled INTEGER DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_depth_events_ts ON depth_events(exchange, symbol, timestamp);
    """)
    conn.close()


def insert_depth_events(conn, exchange, symbol, events):
    conn.executemany(
        "INSERT INTO depth_events (timestamp, exchange, symbol, price, side, event_type, size_before, size_after, size_usd, filled) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [(e["ts"], exchange, symbol, e["price"], e["side"], e["type"],
          e.get("size_before", 0), e.get("size_after", 0), e.get("size_usd", 0), e.get("filled", 0))
         for e in events]
    )
    conn.commit()


def get_depth_events(conn, exchange, symbol, from_ts=None, min_usd=0):
    q = "SELECT timestamp, price, side, event_type, size_before, size_after, size_usd, filled FROM depth_events WHERE exchange=? AND symbol=? "
    params = [exchange, symbol]
    if from_ts:
        q += "AND timestamp >= ? "
        params.append(from_ts)
    if min_usd > 0:
        q += "AND size_usd >= ? "
        params.append(min_usd)
    q += "ORDER BY timestamp DESC LIMIT 200"
    return conn.execute(q, params).fetchall()


def insert_vd(conn, exchange, symbol, records):
    conn.executemany(
        "INSERT OR IGNORE INTO vd (timestamp, exchange, symbol, open, high, low, close, n) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [(r["t"], exchange, symbol, r["o"], r["h"], r["l"], r["c"], r["n"]) for r in records]
    )
    conn.commit()


def insert_stats(conn, exchange, symbol, records):
    conn.executemany(
        "INSERT OR IGNORE INTO stats (timestamp, exchange, symbol, last_price, bid_depth, ask_depth, skew, buy_vol, sell_vol) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                r["t"], exchange, symbol, r["lp"],
                json.dumps(r["bs"]), json.dumps(r["as"]), json.dumps(r["sk"]),
                r.get("vb", 0), r.get("vs", 0),
            )
            for r in records
        ]
    )
    conn.commit()


def insert_candles(conn, exchange, symbol, records):
    conn.executemany(
        "INSERT OR IGNORE INTO candles (timestamp, exchange, symbol, open, high, low, close, buy_vol, sell_vol) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [(r["t"], exchange, symbol, r["o"], r["h"], r["l"], r["c"], r.get("vb", 0), r.get("vs", 0)) for r in records]
    )
    conn.commit()


def insert_custom_depth(conn, exchange, symbol, timestamp, last_price, bid_25, ask_25):
    conn.execute(
        "INSERT OR IGNORE INTO custom_depth (timestamp, exchange, symbol, last_price, bid_depth_25, ask_depth_25) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (timestamp, exchange, symbol, last_price, bid_25, ask_25)
    )
    conn.commit()


def get_custom_depth_history(conn, exchange, symbol, from_ts=None):
    q = "SELECT timestamp, last_price, bid_depth_25, ask_depth_25 FROM custom_depth WHERE exchange=? AND symbol=? "
    params = [exchange, symbol]
    if from_ts:
        q += "AND timestamp >= ? "
        params.append(from_ts)
    q += "ORDER BY timestamp"
    return conn.execute(q, params).fetchall()


def insert_trade(conn, exchange, symbol, timestamp, price, size_usd, side, is_large=0, z_score=0.0):
    conn.execute(
        "INSERT INTO trades (timestamp, exchange, symbol, price, size_usd, side, is_large, z_score) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (timestamp, exchange, symbol, price, size_usd, side, is_large, z_score)
    )
    conn.commit()


def insert_trades_batch(conn, exchange, symbol, trades):
    conn.executemany(
        "INSERT INTO trades (timestamp, exchange, symbol, price, size_usd, side, is_large, z_score) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [(t["ts"], exchange, symbol, t["price"], t["size_usd"], t["side"], t.get("is_large", 0), t.get("z_score", 0.0)) for t in trades]
    )
    conn.commit()


def get_trades_history(conn, exchange, symbol, from_ts=None, large_only=False):
    q = "SELECT timestamp, price, size_usd, side, is_large, z_score FROM trades WHERE exchange=? AND symbol=? "
    params = [exchange, symbol]
    if from_ts:
        q += "AND timestamp >= ? "
        params.append(from_ts)
    if large_only:
        q += "AND is_large = 1 "
    q += "ORDER BY timestamp"
    return conn.execute(q, params).fetchall()


def get_trade_stats(conn, exchange, symbol, from_ts, to_ts):
    """Get trade count and volume per minute bucket."""
    q = """SELECT CAST(timestamp/60 AS INTEGER)*60 as bucket,
           COUNT(*) as count,
           SUM(CASE WHEN side='buy' THEN size_usd ELSE 0 END) as buy_vol,
           SUM(CASE WHEN side='sell' THEN size_usd ELSE 0 END) as sell_vol
           FROM trades WHERE exchange=? AND symbol=? AND timestamp >= ? AND timestamp <= ?
           GROUP BY bucket ORDER BY bucket"""
    return conn.execute(q, (exchange, symbol, from_ts, to_ts)).fetchall()


def get_vd_history(conn, exchange, symbol, from_ts=None):
    q = "SELECT timestamp, close FROM vd WHERE exchange=? AND symbol=? "
    params = [exchange, symbol]
    if from_ts:
        q += "AND timestamp >= ? "
        params.append(from_ts)
    q += "ORDER BY timestamp"
    return conn.execute(q, params).fetchall()


def get_vd_history_multi(conn, exchanges, symbol, from_ts=None):
    """Get VD aggregated across multiple exchanges per timestamp."""
    placeholders = ",".join("?" for _ in exchanges)
    q = f"SELECT timestamp, SUM(close) FROM vd WHERE exchange IN ({placeholders}) AND symbol=? "
    params = list(exchanges) + [symbol]
    if from_ts:
        q += "AND timestamp >= ? "
        params.append(from_ts)
    q += "GROUP BY timestamp ORDER BY timestamp"
    return conn.execute(q, params).fetchall()


def get_stats_history(conn, exchange, symbol, from_ts=None):
    q = "SELECT timestamp, last_price, bid_depth, ask_depth, skew, buy_vol, sell_vol FROM stats WHERE exchange=? AND symbol=? "
    params = [exchange, symbol]
    if from_ts:
        q += "AND timestamp >= ? "
        params.append(from_ts)
    q += "ORDER BY timestamp"
    return conn.execute(q, params).fetchall()


def get_candle_history(conn, exchange, symbol, from_ts=None):
    q = "SELECT timestamp, open, high, low, close, buy_vol, sell_vol FROM candles WHERE exchange=? AND symbol=? "
    params = [exchange, symbol]
    if from_ts:
        q += "AND timestamp >= ? "
        params.append(from_ts)
    q += "ORDER BY timestamp"
    return conn.execute(q, params).fetchall()


def get_candle_history_multi(conn, exchanges, symbol, from_ts=None):
    """Get candle data aggregated across exchanges (volume summed, price averaged)."""
    placeholders = ",".join("?" for _ in exchanges)
    q = f"""SELECT timestamp,
            AVG(open) as open, MAX(high) as high, MIN(low) as low, AVG(close) as close,
            SUM(buy_vol) as buy_vol, SUM(sell_vol) as sell_vol
            FROM candles WHERE exchange IN ({placeholders}) AND symbol=? """
    params = list(exchanges) + [symbol]
    if from_ts:
        q += "AND timestamp >= ? "
        params.append(from_ts)
    q += "GROUP BY timestamp ORDER BY timestamp"
    return conn.execute(q, params).fetchall()


def get_vd_by_exchange(conn, exchanges, symbol, from_ts=None):
    """Get VD per exchange (for breakdown chart)."""
    result = {}
    for exch in exchanges:
        rows = get_vd_history(conn, exch, symbol, from_ts)
        if rows:
            result[exch] = rows
    return result


def get_record_count(conn, table, exchange, symbol):
    return conn.execute(
        f"SELECT COUNT(*) FROM {table} WHERE exchange=? AND symbol=?",
        (exchange, symbol)
    ).fetchone()[0]


init_db()
