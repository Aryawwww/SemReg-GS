"""Inspect an authorized HSSD repository directory without downloading it."""

import argparse

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="hssd/hssd-models")
    parser.add_argument("path", nargs="?", default="")
    args = parser.parse_args()
    api = HfApi()
    entries = api.list_repo_tree(
        args.repo,
        path_in_repo=args.path,
        repo_type="dataset",
        recursive=False,
    )
    for entry in entries:
        size = getattr(entry, "size", "")
        print(f"{type(entry).__name__}\t{entry.path}\t{size}")


if __name__ == "__main__":
    main()
