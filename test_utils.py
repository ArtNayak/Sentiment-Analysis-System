"""Run with: python test_utils.py"""
from utils import extract_key_themes, filter_spam, is_safe_url, is_spam

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
print("utils tests passed")
