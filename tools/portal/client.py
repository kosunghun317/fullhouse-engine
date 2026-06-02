"""Small Supabase REST client used by portal replay tools."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable

from .schema import ANON_KEY, SUPABASE_URL, TABLES

PAGE_SIZE_CAP = 1000


def normalize_page_size(page_size: int) -> int:
    return min(PAGE_SIZE_CAP, max(1, int(page_size)))


def build_url(table: str, params: dict[str, str | int], supabase_url: str = SUPABASE_URL) -> str:
    query = urllib.parse.urlencode(params, safe=",.*()")
    return f"{supabase_url}/rest/v1/{table}?{query}"


def request_json(url: str, token: str, timeout: int, retries: int, anon_key: str = ANON_KEY) -> list[dict]:
    headers = {
        "apikey": anon_key,
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read().decode("utf-8")
            return json.loads(payload)
        except urllib.error.HTTPError as exc:
            try:
                body = exc.read().decode("utf-8")
            except Exception:
                body = ""
            if body and hasattr(exc, "add_note"):
                exc.add_note(body)
            elif body:
                print(body, flush=True)
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(min(8.0, 1.5 * (attempt + 1)))
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(min(8.0, 1.5 * (attempt + 1)))
    raise RuntimeError(f"request failed after {retries + 1} attempts: {url}") from last_error


def fetch_table(
    table: str,
    token: str,
    page_size: int,
    timeout: int,
    retries: int,
    progress: Callable[[str], None] | None = print,
) -> list[dict]:
    spec = TABLES[table]
    rows: list[dict] = []
    offset = 0
    page_size = normalize_page_size(page_size)
    while True:
        params = {
            "select": spec["select"],
            "order": spec["order"],
            "limit": page_size,
            "offset": offset,
        }
        page = request_json(build_url(table, params), token, timeout, retries)
        rows.extend(page)
        if progress:
            progress(f"{table}: fetched {len(rows)} rows")
        if len(page) < page_size:
            return rows
        offset += page_size
