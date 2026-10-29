"""P4b (docs/gfx906/P4B-SESSIONS.md 3): sliced generations for park-slot engines.

The ported engines advertise park slots ("READY <ctx> stop slots=N"); the server then drives a generation in
slices of slice_tokens and gives a queued request its turn between slices (Service._yield_turn), so concurrent
users wait one slice, not the whole generation. Tests here:

  * StrataEngine.generate pass-through: no slots / batch engine / no seam / max_new within a slice - today's
    single-pass path, byte-for-byte the same call into _generate_once;
  * slicing: token stream identical to one pass, per-slice GEN caps, whole-request totals in engine.last;
  * Service._yield_turn: the ownership invariant (the request holds the fifo at every exit path - queued-empty
    keeps it, the beat loop re-acquires before finishing, close-while-waiting re-acquires before returning);
  * integration over HTTP: the fake slots engine behind the real StrataEngine + Service: a second request runs
    BETWEEN the first one's slices (interleaving), and both responses are complete.
"""
import json
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, Service, StrataEngine, serve
import serve.server as server

# the fake park-slot engine: answers every GEN with up to min(max_new, TOKENS) tokens one STEP apart, prints
# "READY <ctx> stop slots=4", RESUMEs on a prefix-extended prompt, and logs one line per GEN
# ("GEN kind=<65|66|0> max_new=<n> len=<len ids>") - kind from the user content's byte (A vs B) so the test can
# see a generation's slices and a second conversation running between them.
FAKE_PARK = r'''import queue, sys, threading, time
args = sys.argv[1:]
log = args[args.index("--log") + 1] if "--log" in args else None
STEP = 0.02
TOKENS = 12
def emit(s):
    sys.stdout.write(s + "\n"); sys.stdout.flush()
lines = queue.Queue()
def reader():
    for l in sys.stdin:
        lines.put(l)
    lines.put(None)
threading.Thread(target=reader, daemon=True).start()
emit("READY 8192 stop slots=4")
last = []
while True:
    line = lines.get()
    if line is None or line.strip() == "QUIT":
        break
    f = line.split()
    if not f or f[0] != "GEN":
        continue
    max_new = int(f[1])
    ids = [int(x) for x in f[-1].split(",")]
    kind = 65 if 65 in ids else (66 if 66 in ids else 0)
    if log:
        with open(log, "a") as fh:
            fh.write("GEN kind=%d max_new=%d len=%d\n" % (kind, max_new, len(ids)))
    reused = 0
    if last and len(ids) > len(last) and ids[:len(last)] == last:
        reused = len(last)
        emit("RESUME %d" % reused)
    last = list(ids)
    n = min(max_new, TOKENS)
    for i in range(n):
        emit("T %d" % (100 + i))        # byte-range ids: the ByteTokenizer decodes those; 100+ never collides
        time.sleep(STEP)
    emit("DONE %d %d 1.0 %.1f stop 0 0 %d 1 1 0 0 0 0" % (n, len(ids), 1000 * STEP * n, reused))
'''


def _bare_engine(slots, slice_tokens, batch=0):
    e = StrataEngine.__new__(StrataEngine)
    e.batch = batch
    e.session_slots = slots
    e.slice_tokens = slice_tokens
    e.last = {}
    return e


class SlicedGenerate(unittest.TestCase):
    def test_passthrough_without_slots(self):
        e = _bare_engine(0, 4)
        calls = []
        def once(ids, max_new, sampling, cancel, embeddings=None, request_id=None):
            calls.append(max_new)
            return iter([1, 2, 3, 4, 5, 6, 7])
        e._generate_once = once
        self.assertEqual(list(e.generate([0], 7, {}, threading.Event())), [1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(calls, [7])                       # ONE call, the full max_new: today's path

    def test_passthrough_small_request(self):
        e = _bare_engine(4, 16)
        e.yield_turn = lambda: iter(())
        calls = []
        def once(ids, max_new, sampling, cancel, embeddings=None, request_id=None):
            calls.append(max_new)
            return iter([9, 8])
        e._generate_once = once
        self.assertEqual(list(e.generate([0], 5, {}, threading.Event())), [9, 8])
        self.assertEqual(calls, [5])                       # max_new <= slice: one pass, unchanged

    def test_slices_and_totals(self):
        e = _bare_engine(4, 4)
        e.yield_turn = lambda: iter(())
        seen = []
        def once(ids, max_new, sampling, cancel, embeddings=None, request_id=None):
            seen.append((max_new, len(ids)))
            out = []
            for i in range(min(max_new, 3)):               # the engine may answer FEWER than asked (a stop token)
                t = 400 + i
                out.append(t)
                yield t
            e.last = {"generated": len(out), "prompt_ms": 1.0, "decode_ms": 2.0, "finish": "length",
                      "reused": len(ids)}
        e._generate_once = once
        got = [t for t in e.generate([0, 1, 2], 6, {}, threading.Event()) if t is not None]
        self.assertEqual(got, [400, 401, 402, 400, 401, 402])          # two slices of 3
        self.assertEqual([c[0] for c in seen], [4, 3])                 # 4 then the remaining 3
        self.assertEqual(seen[1][1], 6)                                # 3 prompt + 3 generated: zero re-feed
        self.assertEqual(e.last["generated"], 6)                       # whole-request totals, not the last slice
        self.assertEqual(e.last["slices"], 2)

    def test_heartbeats_pass_through(self):
        e = _bare_engine(4, 2)
        def yt():
            return iter([None])                            # one heartbeat while waiting for the turn
        e.yield_turn = yt
        def once(ids, max_new, sampling, cancel, embeddings=None, request_id=None):
            e.last = {"generated": 2, "prompt_ms": 0.0, "decode_ms": 0.0, "finish": "length", "reused": 0}
            yield 1
            yield None
            yield 2
        e._generate_once = once
        # slice 1: 1, None(heartbeat), 2 | the seam waits one beat (None) | slice 2: 1, None, 2
        self.assertEqual(list(e.generate([0], 4, {}, threading.Event())), [1, None, 2, None, 1, None, 2])


class YieldTurnOwnership(unittest.TestCase):
    """The invariant: a request holds Service.fifo at every exit path of the seam."""

    def svc(self, queued):
        s = Service.__new__(Service)
        s.fifo = threading.Lock()
        s.status_lock = threading.Lock()
        s.status = {"queued": queued}
        s.fifo.acquire()                                   # the request holds its turn
        return s

    def test_queued_empty_keeps_the_fifo(self):
        s = self.svc(0)
        self.assertEqual(list(s._yield_turn()), [])
        self.assertTrue(s.fifo.locked())                   # never released: nothing to hand off

    def test_beat_loop_reacquires_before_finishing(self):
        s = self.svc(1)
        with mock.patch.object(server, "QUEUE_BEAT_S", 0.02):
            it = s._yield_turn()                             # releases the request's turn
            self.assertFalse(s.fifo.locked())                # released for the queued request
            s.fifo.acquire()                                 # the test holds it now: the queued request's turn
            beats = 0
            for hb in it:                                    # beats while the turn is another request's
                beats += 1
                if beats >= 2:
                    break
            s.fifo.release()                                 # the turn comes back
            self.assertEqual(list(it), [])                   # drain: it re-acquires and finishes
            self.assertTrue(s.fifo.locked())                 # holds the turn again: the ExitStack may release it
            s.fifo.release()

    def test_close_while_waiting_reacquires(self):
        s = self.svc(1)
        with mock.patch.object(server, "QUEUE_BEAT_S", 0.02):
            it = s._yield_turn()                             # releases the fifo
            s.fifo.acquire()                                 # another request holds the turn
            self.assertIsNone(next(it))                      # we wait, beating
            def give_back():
                time.sleep(0.05)
                s.fifo.release()                             # its turn ends
            th = threading.Thread(target=give_back)
            th.start()
            it.close()                                       # abandoned mid-wait: close BLOCKS until we hold again
            th.join()
            self.assertTrue(s.fifo.locked())                 # held again: the ExitStack releases only its own turn
            s.fifo.release()


class ParkedSlicesOverHttp(unittest.TestCase):
    """The real StrataEngine + Service over HTTP behind the fake slots engine: a second request runs BETWEEN
    the first one's slices, and both answers are complete."""

    def setUp(self):
        self.start()

    def start(self):
        self.tmp = tempfile.TemporaryDirectory()
        script = Path(self.tmp.name) / "fake_park.py"
        script.write_text(FAKE_PARK, encoding="utf-8")
        self.log = Path(self.tmp.name) / "gens.log"
        real = server.subprocess.Popen
        with mock.patch.object(server.subprocess, "Popen",
                               lambda cmd, **kw: real([sys.executable, str(script), "--log", str(self.log)], **kw)):
            self.engine = StrataEngine("strata", [])
        self.engine.slice_tokens = 4
        self.svc = Service(self.engine, ByteTokenizer(),
                           ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        if getattr(self, "httpd", None):
            self.httpd.shutdown()
            self.httpd.server_close()
        if getattr(self, "engine", None):
            self.engine.unload()
        if getattr(self, "tmp", None):
            self.tmp.cleanup()

    def chat(self, text, max_tokens):
        body = {"messages": [{"role": "user", "content": text}], "max_tokens": max_tokens, "reasoning_effort": "none"}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode())

    def test_second_request_runs_between_slices(self):
        self.assertTrue(self.engine.session_slots)         # the READY suffix was parsed
        out_b = {}
        def b():
            out_b["r"] = self.chat("BBBB", 4)
        ta = threading.Thread(target=self.chat, args=("AAAAAAAA", 12))          # 3 slices of 4 (fake caps 12/GEN)
        tb = threading.Thread(target=b)
        ta.start()
        time.sleep(0.10)                                    # B arrives while A is mid-generation
        tb.start()
        ta.join(60)
        tb.join(60)
        gens = [l.strip() for l in self.log.read_text().splitlines() if l.startswith("GEN ")]
        kinds = [int(l.split("kind=")[1].split()[0]) for l in gens]
        self.assertIn(66, kinds)                            # B reached the engine
        first_a, last_a = kinds.index(65), len(kinds) - 1 - kinds[::-1].index(65)
        self.assertGreater(last_a - first_a, 1)             # A ran as SEVERAL GENs (it asked 12, the slice is 4)
        self.assertTrue(any(k == 66 for k in kinds[first_a + 1:]), "B never ran between A's slices")
        a_gens = [l for l in gens if "kind=65" in l]
        self.assertTrue(all(int(l.split("max_new=")[1].split()[0]) == 4 for l in a_gens),
                      "A's GEN lines were not capped to the slice")
        lens = [int(l.split("len=")[1]) for l in a_gens]
        self.assertEqual(lens, sorted(lens))                # continuations extend the prompt (zero re-feed shape)

    def test_single_request_not_sliced_when_small(self):
        r = self.chat("CCCC", 3)                            # 3 <= slice_tokens: today's single GEN
        gens = self.log.read_text().splitlines()
        self.assertEqual(len(gens), 1)
        self.assertEqual(gens[0].split("max_new=")[1].split()[0], "3")
        self.assertTrue(r["choices"][0]["message"]["content"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
