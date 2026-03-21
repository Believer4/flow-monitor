import os
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://eu-central-1.mmt.gg"


class MMTClient:
    def __init__(self):
        self.api_key = os.getenv("MMT_API_KEY")
        if not self.api_key:
            raise ValueError("MMT_API_KEY not set")
        self.session = requests.Session()
        self.session.headers["X-API-Key"] = self.api_key

    def get_stats(self, exchange, symbol, tf="1m", from_ts=None, to_ts=None):
        params = {"exchange": exchange, "symbol": symbol, "tf": tf}
        if from_ts:
            params["from"] = from_ts
        if to_ts:
            params["to"] = to_ts
        r = self.session.get(f"{BASE_URL}/api/v1/stats", params=params)
        r.raise_for_status()
        return r.json()

    def get_vd(self, exchange, symbol, tf="1m", bucket=1, from_ts=None, to_ts=None):
        params = {"exchange": exchange, "symbol": symbol, "tf": tf, "bucket": bucket}
        if from_ts:
            params["from"] = from_ts
        if to_ts:
            params["to"] = to_ts
        r = self.session.get(f"{BASE_URL}/api/v1/vd", params=params)
        r.raise_for_status()
        return r.json()

    def get_orderbook(self, exchange, symbol, levels="full"):
        params = {"exchange": exchange, "symbol": symbol, "levels": levels}
        r = self.session.get(f"{BASE_URL}/api/v1/orderbook", params=params)
        r.raise_for_status()
        return r.json()

    def get_candles(self, exchange, symbol, tf="1m", from_ts=None, to_ts=None):
        params = {"exchange": exchange, "symbol": symbol, "tf": tf}
        if from_ts:
            params["from"] = from_ts
        if to_ts:
            params["to"] = to_ts
        r = self.session.get(f"{BASE_URL}/api/v1/candles", params=params)
        r.raise_for_status()
        return r.json()
