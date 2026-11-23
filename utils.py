"""Pure helper functions for the app (no Streamlit or model imports, so they are easy to test)."""
import ipaddress
import re
import socket
from urllib.parse import urlparse

from sklearn.feature_extraction.text import CountVectorizer

SPAM_PHRASES = ("click here", "buy now", "free money", "limited offer", "act now",
                "earn money", "work from home", "visit my", "whatsapp me", "subscribe to my")


def is_safe_url(url: str) -> tuple[bool, str]:
    """Allow only public http(s) URLs. Blocks localhost, private networks and cloud-metadata
    addresses so the scraper cannot be pointed at the server's own network."""
    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False, "Enter a full URL starting with http:// or https://"
    try:
        addresses = {info[4][0] for info in socket.getaddrinfo(parsed.hostname, None)}
    except socket.gaierror:
        return False, "Could not resolve that host name."
    for address in addresses:
        ip = ipaddress.ip_address(address.split("%")[0])
        if not ip.is_global:  # loopback, private, link-local (169.254.x.x metadata), reserved...
            return False, "That address is not a public website."
    return True, ""


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
