import httpx
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


def test_single_word_verse_ar_rahman_is_matched_and_playable(client):
    for text in ("ٱلرَّحْمَـٰنُ", "الرحمن"):
        r = client.post("/api/match", json={"text": text, "asr_confidence": 0.9}).json()
        assert r["status"] == "matched" and r["range"]["ref"] == "55:1"
        assert r["play"]["segments"] and not r.get("forced")


# --- WebSocket API (Flutter) ------------------------------------------------

def _ws_ask(ws, **msg):
    ws.send_json(msg)
    return ws.receive_json()


def test_ws_result_matches_http_match(client):
    texts = ["قل هو الله احد الله الصمد", "الله لا اله الا هو الحي القيوم", "الحمد لله",
             "قطة تأكل سمكة كبيرة جدا", "الرحمن"]
    with client.websocket_connect("/ws/recite") as ws:
        for text in texts:
            ws_r = _ws_ask(ws, type="utterance", text=text, id=text)
            ws.send_json({"type": "reset"}); assert ws.receive_json()["type"] == "reset"
            http_r = client.post("/api/match", json={"text": text}).json()
            assert ws_r.pop("type") == "result" and ws_r.pop("id") == text
            for r in (ws_r, http_r):
                r.pop("timings_ms")
                for seg in (r.get("play") or {}).get("segments", []):
                    seg["audio_url"] = seg["audio_url"].rsplit("/", 1)[-1]
            assert ws_r == http_r


def test_ws_audio_urls_point_at_this_server(client):
    with client.websocket_connect("/ws/recite") as ws:
        r = _ws_ask(ws, type="utterance", text="قل هو الله احد الله الصمد")
    urls = [s["audio_url"] for s in r["play"]["segments"]]
    assert urls == ["http://testserver/api/audio/112001.mp3", "http://testserver/api/audio/112002.mp3"]


def test_ws_session_follows_forward_and_reset_clears_it(client):
    refrain = "فبأي آلاء ربكما تكذبان"
    with client.websocket_connect("/ws/recite") as ws:
        first = _ws_ask(ws, type="utterance", text=refrain)
        second = _ws_ask(ws, type="utterance", text=refrain)
        assert first["range"]["end_gid"] < second["range"]["start_gid"]     # moved on, not stuck
        ws.send_json({"type": "reset"}); ws.receive_json()
        assert _ws_ask(ws, type="utterance", text=refrain)["range"] == first["range"]


def test_ws_play_and_errors_keep_the_connection_open(client):
    with client.websocket_connect("/ws/recite") as ws:
        assert _ws_ask(ws, type="nope", id=7) == {"type": "error", "message": "unknown type 'nope'", "id": 7}
        assert _ws_ask(ws, type="play", start_gid=10, end_gid=5)["type"] == "error"
        assert _ws_ask(ws, type="utterance")["type"] == "result"          # missing text is just empty input
        ws.send_text("not json")
        assert ws.receive_json()["type"] == "error"
        assert _ws_ask(ws, type="play", start_gid=0, end_gid=0)["play"]["segments"]


def test_audio_proxy_rejects_anything_but_ayah_files(client):
    for bad in ("..%2Fetc%2Fpasswd", "abc.mp3", "115001.mp3", "001000.mp3", "1001.mp3"):
        assert client.get(f"/api/audio/{bad}").status_code in (400, 404)


def test_audio_proxy_streams_origin_with_range(client, monkeypatch):
    seen = {}

    def handler(request):
        seen["url"], seen["range"] = str(request.url), request.headers.get("range")
        return httpx.Response(206, content=b"ID3abc", headers={"content-type": "audio/mpeg",
                                                               "content-range": "bytes 0-5/100"})
    real = httpx.AsyncClient
    monkeypatch.setattr(qari_demo.httpx, "AsyncClient",
                        lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    r = client.get("/api/audio/112001.mp3", headers={"range": "bytes=0-5"})
    assert r.status_code == 206 and r.content == b"ID3abc" and r.headers["content-range"] == "bytes 0-5/100"
    assert seen == {"url": "https://everyayah.com/data/Alafasy_128kbps/112001.mp3", "range": "bytes=0-5"}
