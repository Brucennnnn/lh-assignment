"""Interactive demo.

    python main.py                # ask questions against the real provider
    python main.py --offline      # deterministic stub, no API key needed
    python main.py --rebuild      # re-ingest and re-embed the knowledge base
"""
import argparse

from task1.src import llm
from task1.src.agent import Agent
from task1.src import config, ingestion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="use the deterministic stub")
    parser.add_argument("--rebuild", action="store_true", help="rebuild the vector index")
    args = parser.parse_args()

    if args.offline:
        llm.use_stub()
        config.apply_stub_thresholds()
        store = ingestion.build_index(persist=False)
    else:
        store = ingestion.load_index(rebuild=args.rebuild)

    bot = Agent(store)
    print(f"LH Bank internal knowledge assistant ({len(store.chunks)} chunks indexed). "
          f"Ctrl-C or 'exit' to quit.\n")

    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if query.lower() in {"exit", "quit"}:
            return
        if not query:
            continue

        result = bot.answer(query)
        print(f"\n{result['answer']}\n")
        if result["sources"]:
            print("Sources:")
            for s in result["sources"]:
                print(f"- {s['title']} — {s['source']}")
            print()
        print(f"[{result['status']} | request_id={result['request_id']} "
              f"| confidence={result['confidence']}]\n")


if __name__ == "__main__":
    main()
