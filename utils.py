"""Pure helper functions for the app (no Streamlit or model imports, so they are easy to test)."""
import ipaddress
import re
import socket
from dataclasses import dataclass
from urllib.parse import urlparse, urlunparse

from sklearn.feature_extraction.text import CountVectorizer

SPAM_PHRASES = ("click here", "buy now", "free money", "limited offer", "act now",
                "earn money", "work from home", "visit my", "whatsapp me", "subscribe to my")


_BAD_URL_CHARS = re.compile("[%s]" % re.escape("".join(map(chr, [*range(33), 127, 92]))))  # control chars, whitespace, backslash
ALLOWED_PORTS = (None, 80, 443, 8080, 8443)


@dataclass(frozen=True)
class ScrapeTarget:
    url: str   # normalized URL the browser must open (this exact string is passed to page.goto)
    host: str  # the only host the browser may talk to
    ip: str    # the public address validated for that host; the browser is pinned to it


def resolve_public_target(url: str) -> tuple[ScrapeTarget | None, str]:
    """Validate a user-supplied URL for scraping and resolve it ONCE.

    Returns (target, "") or (None, reason). The caller must make the browser connect to
    `target.ip` for `target.host` (see scrape_comments) so that a second DNS answer cannot
    point it somewhere private (DNS rebinding), and must open `target.url`, not the raw input.
    URLs are rejected if different parsers could read them differently: whitespace, control
    characters, backslashes and user-info ("user@host") are all refused.
    """
    raw = url.strip()
    if _BAD_URL_CHARS.search(raw):
        return None, "That URL contains characters that are not allowed."
    parsed = urlparse(raw)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return None, "Enter a full URL starting with http:// or https://"
    if "@" in parsed.netloc:
        return None, "URLs with a user name or password are not allowed."
    try:
        port = parsed.port
    except ValueError:
        return None, "That URL has an invalid port."
    if port not in ALLOWED_PORTS:
        return None, "Only ports 80, 443, 8080 and 8443 are allowed."

    host = parsed.hostname.lower().rstrip(".")
    try:
        infos = socket.getaddrinfo(host, port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)
    except socket.gaierror:
        return None, "Could not resolve that host name."
    addresses = [info[4][0].split("%")[0] for info in infos]
    if not addresses or not all(ipaddress.ip_address(a).is_global for a in addresses):
        # loopback, private, link-local (169.254.x.x metadata), reserved...
        return None, "That address is not a public website."
    ip = next((a for a in addresses if ":" not in a), addresses[0])  # prefer IPv4

    netloc = f"[{host}]" if ":" in host else host
    if port:
        netloc += f":{port}"
    normalized = urlunparse((parsed.scheme, netloc, parsed.path or "/", "", parsed.query, ""))
    return ScrapeTarget(normalized, host, ip), ""


def is_safe_url(url: str) -> tuple[bool, str]:
    target, reason = resolve_public_target(url)
    return target is not None, reason


def request_allowed(request_url: str, allowed_host: str) -> bool:
    """Used to filter every request the scraping browser makes (navigations, redirects,
    scripts, images...). Only the validated host is reachable, so a redirect or a script cannot
    send the browser to another host, public or private. Inline data:/blob: URLs are harmless."""
    parsed = urlparse(request_url)
    if parsed.scheme in ("data", "blob", "about"):
        return True
    return parsed.scheme in ("http", "https") and (parsed.hostname or "").lower().rstrip(".") == allowed_host


def is_spam(comment: str) -> bool:
    lowered = comment.lower()
    if len(re.findall(r"https?://|www\.", lowered)) >= 2:
        return True
    if any(phrase in lowered for phrase in SPAM_PHRASES):
        return True
    if re.search(r"(.)\1{9,}", comment):  # the same character repeated 10+ times
        return True
    letters = sum(ch.isalpha() for ch in comment)
    return len(comment) >= 10 and letters / len(comment) < 0.3  # mostly digits/symbols


def filter_spam(comments: list[str]) -> tuple[list[str], int]:
    """Drop spam-looking comments and exact (case-insensitive) duplicates. Returns (kept, removed_count)."""
    seen, kept = set(), []
    for comment in comments:
        key = " ".join(comment.lower().split())
        if key in seen or is_spam(comment):
            continue
        seen.add(key)
        kept.append(comment)
    return kept, len(comments) - len(kept)


def extract_key_themes(text: str, num_themes: int = 5) -> list[str]:
    """Most frequent words and two-word phrases. Returns [] if the text has no usable words."""
    text = re.sub(r"http\S+|@\S+|[^A-Za-z\s]", "", text.lower())
    try:
        vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words="english", max_features=num_themes)
        vectorizer.fit_transform([text])
    except ValueError:  # empty vocabulary: only stop words or symbols
        return []
    return list(vectorizer.get_feature_names_out())
