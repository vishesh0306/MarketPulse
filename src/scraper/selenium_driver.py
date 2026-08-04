"""Headless Chrome session management: randomized UA/viewport, context-managed get_driver()."""

from __future__ import annotations

import random
from contextlib import contextmanager
from typing import Iterator

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.remote.remote_connection import RemoteConnection
from webdriver_manager.chrome import ChromeDriverManager

from src.scraper.anti_detection import random_user_agent

_VIEWPORTS = [(1366, 768), (1440, 900), (1536, 864), (1920, 1080), (1280, 720)]
_PAGE_LOAD_TIMEOUT_SECONDS = 25


def random_viewport() -> tuple[int, int]:
    """Returns a randomized, realistic desktop viewport (width, height)."""
    return random.choice(_VIEWPORTS)


def resolve_driver_path() -> str:
    """Downloads/verifies the chromedriver binary and returns its local path.

    Meant to be called once in the parent process before spawning a worker pool — see
    get_driver()'s driver_path parameter for why.
    """
    return ChromeDriverManager().install()


@contextmanager
def get_driver(
    headless: bool = True, user_agent: str | None = None, driver_path: str | None = None
) -> Iterator[WebDriver]:
    """Yields a configured Chrome WebDriver (randomized user-agent/viewport) and quits it on exit.

    Uses an "eager" page-load strategy plus a hard page-load timeout and disabled image
    loading: we only need DOM text, and waiting on every image/media subresource from a
    third-party host is what caused observed multi-minute hangs during real collection runs.

    A global socket-level timeout on the webdriver connection (RemoteConnection.set_timeout)
    is also set: page_load_timeout only bounds navigation, but a hang can also occur inside
    chromedriver itself while it waits on browser IPC during an ordinary command like
    find_elements — observed in practice as a second, longer hang surviving the first fix.

    driver_path should be pre-resolved via resolve_driver_path() and passed in when multiple
    get_driver() calls may run concurrently (one per worker process): ChromeDriverManager's
    own cache-file handling isn't safe under concurrent first-touch access — observed in
    practice as an intermittent "chromedriver.exe executable may have wrong permissions"
    error when several hashtags started scraping in the same instant.
    """
    RemoteConnection.set_timeout(_PAGE_LOAD_TIMEOUT_SECONDS)
    width, height = random_viewport()
    ua = user_agent or random_user_agent()

    options = Options()
    options.page_load_strategy = "eager"
    if headless:
        options.add_argument("--headless=new")
    options.add_argument(f"--window-size={width},{height}")
    options.add_argument(f"--user-agent={ua}")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    # Chrome starts several background services on launch that this scraper never uses
    # (push-notification/GCM registration, component update checks, sync, USB device
    # probing) — they fail harmlessly against endpoints we never call, but flood stderr
    # with unrelated "ERROR:" noise that's easy to mistake for a real scraping problem.
    options.add_argument("--disable-background-networking")
    options.add_argument("--disable-component-update")
    options.add_argument("--disable-sync")
    options.add_argument("--log-level=3")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_experimental_option("prefs", {"profile.managed_default_content_settings.images": 2})

    service = Service(driver_path or resolve_driver_path())
    driver = webdriver.Chrome(service=service, options=options)
    try:
        driver.set_page_load_timeout(_PAGE_LOAD_TIMEOUT_SECONDS)
        driver.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument",
            {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"},
        )
        yield driver
    finally:
        driver.quit()
