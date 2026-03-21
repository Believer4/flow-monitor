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
    """)
    conn.close()


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


def get_vd_history(conn, exchange, symbol, from_ts=None):
    q = "SELECT timestamp, close FROM vd WHERE exchange=? AND symbol=? "
    params = [exchange, symbol]
    if from_ts:
        q += "AND timestamp >= ? "
        params.append(from_ts)
    q += "ORDER BY timestamp"
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


def get_record_count(conn, table, exchange, symbol):
    return conn.execute(
        f"SELECT COUNT(*) FROM {table} WHERE exchange=? AND symbol=?",
        (exchange, symbol)
    ).fetchone()[0]


init_db()
