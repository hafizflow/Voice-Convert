import pytest
from fastapi.testclient import TestClient

from tools import qari_demo

pytestmark = pytest.mark.skipif(
    not (qari_demo.ROOT / "data" / "timestamps" / "afasy.json").exists(),
    reason="run python -m tools.build_timestamps first")


@pytest.fixture(scope="module")
def client():
    return TestClient(qari_demo.app)


def test_match_returns_text_segments_and_range(client):
    r = client.post("/api/match", json={"text": "قل هو الله احد الله الصمد"}).json()
    assert r["status"] == "matched" and r["range"]["ref"] == "112:1 to 112:2"
    assert [s["label"] for s in r["play"]["segments"]] == ["112:1", "112:2"]
    assert r["play"]["segments"][0]["audio_url"].endswith("/112001.mp3")
    assert r["range"]["text"].startswith("قُلْ")      # Tanzil text, never the input


def test_partial_range_has_padded_start_and_open_end_at_ayah_end(client):
    r = client.post("/api/match", json={"text": "لا تاخذه سنه ولا نوم له ما في السماوات وما في الارض", "pad_ms": 100}).json()
    assert r["status"] == "matched"
    seg = r["play"]["segments"][0]
    assert seg["start_ms"] >= 0 and seg["label"] == "2:255"
    assert any(not w["in_range"] for w in seg["words"]) or seg["start_ms"] > 0   # a sub-range, not the whole ayah


def test_phrase_shared_by_two_verses_is_ambiguous(client):
    # "الله لا اله الا هو الحي القيوم" opens both 2:255 and 3:2
    r = client.post("/api/match", json={"text": "الله لا اله الا هو الحي القيوم"}).json()
    assert r["status"] == "ambiguous"
    assert r["forced"]["reason"] == "ambiguous" and r["range"]["ref"].startswith("2:255")   # earliest of the tie


def test_ambiguous_refrain_still_plays_the_best_guess_and_lists_the_rest(client):
    r = client.post("/api/match", json={"text": "فبأي آلاء ربكما تكذبان"}).json()
    assert r["status"] == "ambiguous" and len(r["alternatives"]) >= 2
    assert r["range"] is not None and r["play"]["segments"] and r["forced"]["candidates"] >= 2


def test_best_guess_follows_the_session_forward(client):
    from qari.text.index import WordIndex
    idx = WordIndex.from_file()
    end = idx.ayah_range(55, 15)[1]
    r = client.post("/api/match", json={"text": "فبأي آلاء ربكما تكذبان", "last_end_gid": end}).json()
    assert r["range"]["ref"].startswith("55:16")


def test_too_short_input_plays_the_top_candidate(client):
    r = client.post("/api/match", json={"text": "الحمد لله"}).json()
    assert r["status"] == "too_short" and r["forced"]["reason"] == "too_short"
    assert r["range"]["ref"].startswith("1:2")


def test_gibberish_with_no_candidate_has_nothing_to_play(client):
    r = client.post("/api/match", json={"text": "قطة تأكل سمكة كبيرة جدا"}).json()
    assert r["range"] is None and "forced" not in r


def test_play_endpoint_rejects_bad_range(client):
    assert client.get("/api/play?start_gid=10&end_gid=5").status_code == 400
