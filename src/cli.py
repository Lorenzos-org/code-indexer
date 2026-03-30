"""CLI: index / query / daemon / session / stats / log-conversation."""
from __future__ import annotations
import argparse
import json
import signal
import sys
import time
import threading
from pathlib import Path

from .config import load_config
from .indexer import CodeIndexer


# ── formatters ─────────────────────────────────────────────────────────────

def print_results(results: list, max_chars: int = 600):
    for i, r in enumerate(results, 1):
        m = r["metadata"]
        print(f"\n{'─'*64}")
        print(f"#{i}  score={r['score']:.3f}  [{m.get('language','?')}]  "
              f"{m.get('file_path','?')}  "
              f"L{m.get('start_line','?')}–{m.get('end_line','?')}")
        if m.get("enclosing_context"):
            print(f"    context: {m['enclosing_context']}")
        print()
        for line in r["content"][:max_chars].splitlines():
            print(f"    {line}")
        if len(r["content"]) > max_chars:
            print("    …")


# ── command handlers ────────────────────────────────────────────────────────

def cmd_index(args, ix: CodeIndexer):
    if args.full:
        print("Forcing full re-index (ignoring hashes)…")
        ix.store._state.clear()
        ix.store._save_state()

    stats = ix.index(args.path, incremental=not args.full)
    print(json.dumps(stats, indent=2))

    if args.watch:
        _watch(args.path, ix)


def cmd_query(args, ix: CodeIndexer):
    q = " ".join(args.query)
    if args.sessions:
        res = ix.query_sessions(q, n=args.n)
    else:
        res = ix.query_code(q, n=args.n, language=args.lang, path_contains=args.path)

    print(f"\nQuery : {res['query']}")
    print(f"Time  : {res['duration_ms']}ms  |  Results: {len(res['results'])}")
    print_results(res["results"])


def cmd_daemon(args, ix: CodeIndexer):
    paths    = args.paths
    interval = args.interval

    print(f"Daemon started (interval={interval}s)")
    print(f"Watching: {paths}")

    running = True

    def stop(sig, frame):
        nonlocal running
        running = False
        print("\nShutting down…")

    signal.signal(signal.SIGINT,  stop)
    signal.signal(signal.SIGTERM, stop)

    while running:
        for p in paths:
            try:
                s = ix.index(p, incremental=True)
                if s["files_indexed"] > 0:
                    ts = time.strftime("%H:%M:%S")
                    print(f"[{ts}] {p} +{s['files_indexed']} files  +{s['chunks_created']} chunks")
            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}] Error: {e}")

        for _ in range(interval):
            if not running:
                break
            time.sleep(1)

    print("Daemon stopped.")


def cmd_stats(args, ix: CodeIndexer):
    print(json.dumps(ix.stats, indent=2))


def cmd_session(args, ix: CodeIndexer):
    if args.list:
        sessions = ix.sessions.list_sessions(args.limit)
        print(f"\n{'ID':<35}  {'Events':>6}  {'Last activity'}")
        print("─" * 65)
        for s in sessions:
            print(f"{s['id']:<35}  {s['events']:>6}  {s['last_activity']}")
    elif args.id:
        events = ix.sessions.get_session(args.id)
        print(json.dumps(events, indent=2))
    else:
        print("Use --list or --id SESSION_ID")


def cmd_log(args, ix: CodeIndexer):
    """Log a Q&A pair so it's searchable in the sessions collection."""
    ix.sessions.start_session()
    ix.log_conversation(args.query, args.response, tag=args.tag)
    print(f"Logged to session {ix.sessions.session_id}")


# ── file watcher ─────────────────────────────────────────────────────────────

def _watch(path: str, ix: CodeIndexer):
    """Debounced watchdog watcher."""
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
    except ImportError:
        print("[warn] watchdog not installed — run: pip install watchdog")
        return

    pending: dict = {}
    lock = threading.Lock()

    def drain():
        while True:
            time.sleep(1)
            now = time.time()
            with lock:
                ready = [p for p, ts in pending.items() if now - ts > 2.0]
                for p in ready:
                    del pending[p]
            for p in ready:
                ix.index_file(p)

    threading.Thread(target=drain, daemon=True).start()

    class Handler(FileSystemEventHandler):
        def _enqueue(self, event):
            if not event.is_directory:
                with lock:
                    pending[event.src_path] = time.time()

        def on_modified(self, e): self._enqueue(e)
        def on_created(self, e):  self._enqueue(e)
        def on_deleted(self, e):
            if not e.is_directory:
                ix.remove_file(e.src_path)

    obs = Observer()
    obs.schedule(Handler(), path, recursive=True)
    obs.start()
    print(f"Watching {path}… (Ctrl+C to stop)")
    try:
        while True:
            time.sleep(30)
    except KeyboardInterrupt:
        obs.stop()
    obs.join()


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="code-indexer",
        description="Local semantic code search — Ollama + ChromaDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-c", "--config", help="Path to config.yaml")
    sub = parser.add_subparsers(dest="command")

    # index
    p = sub.add_parser("index", help="Index a directory")
    p.add_argument("path")
    p.add_argument("--full",  action="store_true", help="Force full re-index")
    p.add_argument("--watch", action="store_true", help="Keep watching after initial index")

    # query
    p = sub.add_parser("query", help="Semantic search")
    p.add_argument("query", nargs="+")
    p.add_argument("-n",         type=int, default=5)
    p.add_argument("--lang",     help="Filter by language (e.g. python)")
    p.add_argument("--path",     help="Filter by path substring")
    p.add_argument("--sessions", action="store_true", help="Search session logs instead")

    # daemon
    p = sub.add_parser("daemon", help="Background polling daemon")
    p.add_argument("paths", nargs="+")
    p.add_argument("--interval", type=int, default=300)

    # stats
    sub.add_parser("stats", help="Show index statistics")

    # session
    p = sub.add_parser("session", help="Browse session logs")
    p.add_argument("--list",  action="store_true")
    p.add_argument("--id",    help="Show specific session")
    p.add_argument("--limit", type=int, default=20)

    # log
    p = sub.add_parser("log", help="Record a Q&A turn into sessions collection")
    p.add_argument("query")
    p.add_argument("response")
    p.add_argument("--tag", help="Optional label")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    config  = load_config(args.config)
    indexer = CodeIndexer(config)
    indexer.sessions.start_session()

    dispatch = {
        "index":   cmd_index,
        "query":   cmd_query,
        "daemon":  cmd_daemon,
        "stats":   cmd_stats,
        "session": cmd_session,
        "log":     cmd_log,
    }
    dispatch[args.command](args, indexer)


if __name__ == "__main__":
    main()