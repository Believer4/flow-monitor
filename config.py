SYMBOL = "flow/usd"
BASE_URL = "https://eu-central-1.mmt.gg"

# All spot exchanges with FLOW
EXCHANGES = {
    "binance":  {"tick_size": 0.001},
    "bybit":    {"tick_size": 0.0001},
    "coinbase": {"tick_size": 0.001},
    "okx":      {"tick_size": 0.0001},
}

# Primary exchange (most liquid — used for depth WS)
PRIMARY_EXCHANGE = "binance"

# Backwards compat
EXCHANGE = "binance"
TICK_SIZE = 0.001

# Depth levels returned by stats endpoint (% from mid)
DEPTH_LEVELS = [0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]

# History
HISTORY_DAYS = 90
