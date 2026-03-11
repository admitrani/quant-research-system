import requests
import time
import logging
import os
from dotenv import load_dotenv

load_dotenv()

# For use in production
API_KEY = os.getenv("BINANCE_API_KEY")
SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")


logger = logging.getLogger(__name__)

class APIClient:

    def __init__(self, base_url, max_retries=5, backoff_seconds=1.0, timeout=30):
        self.base_url = base_url
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.timeout = timeout

    def get(self, endpoint, params=None):
        for attempt in range(self.max_retries):
            try:
                response = requests.get(f"{self.base_url}{endpoint}", params=params, timeout=self.timeout)
                response.raise_for_status()
                return response.json()
            except requests.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt == self.max_retries - 1:
                    raise
                wait = self.backoff_seconds * (2 ** attempt)
                logger.info(f"Waiting {wait:.1f}s...")
                time.sleep(wait)
