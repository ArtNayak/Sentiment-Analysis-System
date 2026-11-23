"""Run with: python test_utils.py"""
from utils import extract_key_themes, filter_spam, is_safe_url, is_spam, request_allowed, resolve_public_target

# URL safety: only public http(s) sites
assert not is_safe_url("")[0]
assert not is_safe_url("ftp://example.com")[0]
assert not is_safe_url("file:///etc/passwd")[0]
assert not is_safe_url("http://localhost:8501")[0]
assert not is_safe_url("http://127.0.0.1")[0]
assert not is_safe_url("http://169.254.169.254/latest/meta-data")[0]  # cloud metadata
assert not is_safe_url("http://10.0.0.5")[0]
assert not is_safe_url("http://192.168.1.1")[0]
assert is_safe_url("https://8.8.8.8")[0]

# spam filter keeps real comments, drops spam and duplicates
good = "The proposal has some major flaws in how it treats small businesses."
spam = "CLICK HERE to buy now!!! www.a.com http://b.com"
kept, removed = filter_spam([good, spam, good.upper(), "1234567890 !!!! ???? 0000", "Great initiative, fully support it."])
assert kept == [good, "Great initiative, fully support it."], kept
assert removed == 3
assert is_spam("aaaaaaaaaaaaaaaaa")
assert not is_spam("Short but fine.")

# themes: never crash on empty input
assert extract_key_themes("") == []
assert extract_key_themes("the and of to") == []
assert "policy" in " ".join(extract_key_themes("The policy helps farmers. This policy is good for farmers."))
# --- hardened URL parsing (all of these use IP literals, so no DNS is needed) ---
BACKSLASH = chr(92)
for bad in [
    "http://8.8.8.8@127.0.0.1/",                 # user-info trick: parsers disagree on the host
    "https://user:pw@example.com/",
    "http://127.0.0.1" + BACKSLASH + "@8.8.8.8/",  # backslash confuses some parsers
    "http://8.8.8.8/ x",                         # whitespace
    "http://8.8.8.8/" + chr(10) + "Host: evil",   # line break
    "http://8.8.8.8/" + chr(0),                  # control character
    "http://8.8.8.8:22/",                        # port not allowed
    "http://8.8.8.8:99999/",                     # invalid port
    "http://[::1]/",                             # IPv6 loopback
    "http://[::ffff:127.0.0.1]/",                # IPv4-mapped loopback
    "http://2130706433/",                        # 127.0.0.1 written as a decimal number
    "http://0.0.0.0/",
]:
    assert resolve_public_target(bad)[0] is None, bad

ok, reason = resolve_public_target("HTTP://8.8.8.8/path?q=1#fragment")
assert ok is not None and reason == ""
assert (ok.url, ok.host, ok.ip) == ("http://8.8.8.8/path?q=1", "8.8.8.8", "8.8.8.8"), ok   # normalized, fragment dropped
assert resolve_public_target("http://8.8.8.8:8080/x")[0] is not None
assert resolve_public_target("http://[2606:4700:4700::1111]/")[0].ip == "2606:4700:4700::1111"

# --- the per-request guard used inside the browser ---
assert request_allowed("https://example.com/a.js", "example.com")
assert request_allowed("https://EXAMPLE.com./x", "example.com")          # case and trailing dot
assert request_allowed("data:text/plain;base64,AAAA", "example.com")
assert not request_allowed("https://cdn.example.com/x", "example.com")    # even sub-domains are blocked
assert not request_allowed("http://127.0.0.1:8501/", "example.com")       # redirect to localhost
assert not request_allowed("http://169.254.169.254/latest/meta-data", "example.com")
assert not request_allowed("https://example.com.evil.net/", "example.com")
assert not request_allowed("file:///etc/passwd", "example.com")
assert not request_allowed("ftp://example.com/", "example.com")

print("utils tests passed")
