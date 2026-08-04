"""Headless Chrome session management: randomized UA/viewport, context-managed get_driver()."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from selenium.webdriver.chrome.webdriver import WebDriver


@contextmanager
def get_driver(headless: bool = True) -> Iterator[WebDriver]:
    """Yields a configured Chrome WebDriver (randomized user-agent/viewport) and quits it on exit."""
    raise NotImplementedError
