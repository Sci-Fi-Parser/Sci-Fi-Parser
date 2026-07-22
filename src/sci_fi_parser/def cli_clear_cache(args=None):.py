def cli_clear_cache(args=None):
    parser = argparse.ArgumentParser(prog="clear-cache", description="Clear cache of handled PDFs.")
    parser.add_argument("dir", help="directory containing cache; by default in output directory")
    parser.parse_args()

    if not Path(args.dir).exists():
        print("invalid directory")
        return

    if not list(Path(args.dir).glob("cache.db")):
        print(f"no cache.db found in {args.dir}")
        return

    print("Clearing cache...")
    with Cache(args.dir) as conn:
        conn.clear()
    print("\nFinished.")

    return parser.parse_args()
