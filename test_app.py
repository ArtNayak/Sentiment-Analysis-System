"""UI smoke test: python test_app.py (runs app.py with stand-in models, so nothing is downloaded)."""
import sys
import torch
import transformers
from streamlit.testing.v1 import AppTest

import os
APP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
sys.path.insert(0, os.path.dirname(APP))


class FakeTok:
    def __call__(self, text, **kw):
        return {"input_ids": torch.ones(1, 4, dtype=torch.long)}


class FakeOut:
    logits = torch.tensor([[0.1, 0.2, 2.0]])  # negative, neutral, positive


class FakeModel:
    def __call__(self, **kw):
        return FakeOut()


calls = []


def fake_summarizer(text, **kw):
    calls.append((len(text.split()), kw))
    return [{"summary_text": "short summary"}]


transformers.AutoTokenizer.from_pretrained = lambda *a, **k: FakeTok()
transformers.AutoModelForSequenceClassification.from_pretrained = lambda *a, **k: FakeModel()
transformers.pipeline = lambda *a, **k: fake_summarizer


def run(page=None):
    at = AppTest.from_file(APP, default_timeout=120)
    if page:
        at.session_state["page"] = page
    return at.run()


def no_errors(at, label):
    assert not at.exception, f"{label}: {[e.value for e in at.exception]}"
    print("ok:", label)


# 1. dashboard
at = run()
no_errors(at, "dashboard renders")

# 2. sentiment analysis with spam + duplicates
at = run("Sentiment Analysis")
no_errors(at, "analysis page renders")
text = "This is a fantastic initiative for farmers.\nthis is a fantastic initiative for farmers.\nCLICK HERE to buy now http://a.com www.b.com\nThe proposal has some major flaws in its funding plan."
at.text_area[0].set_value(text)
[b for b in at.button if b.label == "Analyze Sentiment"][0].click()
at.run()
no_errors(at, "analysis with spam text")
assert any("Removed 2" in i.value for i in at.info), [i.value for i in at.info]
assert any("Positive" in m.value for m in at.markdown), "overall sentiment not shown"
print("ok: spam + duplicate removed, result shown")

# 3. input with no usable words (used to crash theme extraction / word cloud)
at = run("Sentiment Analysis")
at.text_area[0].set_value("the and of to a in it is")
[b for b in at.button if b.label == "Analyze Sentiment"][0].click()
at.run()
no_errors(at, "stop-word-only input")

# 4. unsafe scrape URL is refused before any browser starts
at = run("Sentiment Analysis")
at.radio[0].set_value("Scrape from URL")
at.run()
at.text_input[0].set_value("http://localhost:8501")
[b for b in at.button if b.label == "Scrape Comments"][0].click()
at.run()
no_errors(at, "unsafe url")
assert any("not a public website" in e.value for e in at.error), [e.value for e in at.error]
print("ok: localhost scrape refused")

# 5. summary: long text is chunked and truncated; min >= max is refused
at = run("Smart Summaries")
no_errors(at, "summary page renders")
long_text = " ".join(["word"] * 1300)
at.text_area[0].set_value(long_text)
[b for b in at.button if b.label == "Generate Summary"][0].click()
at.run()
no_errors(at, "long summary")
assert len(calls) == 4, calls  # 3 chunks (500, 500, 300) + 1 second pass
assert all(kw["truncation"] is True for _, kw in calls)
print("ok: 1300 words -> 3 chunk calls + 1 final pass, truncation on")

calls.clear()
at = run("Smart Summaries")
at.text_area[0].set_value(long_text)
at.slider[0].set_value(200)
at.slider[1].set_value(100)
[b for b in at.button if b.label == "Generate Summary"][0].click()
at.run()
assert any("smaller than the maximum" in w.value for w in at.warning), [w.value for w in at.warning]
assert not calls
print("ok: min >= max refused, model not called")
print("ALL APP CHECKS PASSED")
