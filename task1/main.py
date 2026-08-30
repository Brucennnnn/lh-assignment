"""Interactive demo.

    python main.py                # ask questions against the knowledge base
    python main.py --rebuild      # re-ingest and re-embed the knowledge base

Needs OPENAI_API_KEY. Copy .env.example to .env first.
"""
import argparse

from src.agent import Agent
from src import ingestion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="rebuild the vector index")
    args = parser.parse_args()

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
