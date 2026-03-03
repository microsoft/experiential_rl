#!/usr/bin/env python3

import argparse
import gzip
import json
import os
import tarfile
from pathlib import Path

from tqdm import tqdm


def extract_wikipedia_corpus(wiki_dir: str, remove_gz: bool = True):
    wiki_file = os.path.join(wiki_dir, "wiki-18.jsonl")
    wiki_gz_file = os.path.join(wiki_dir, "wiki-18.jsonl.gz")

    if os.path.exists(wiki_file):
        print(f"Wikipedia corpus already exists at {wiki_file}")
        return wiki_file

    if not os.path.exists(wiki_gz_file):
        print(f"Missing {wiki_gz_file}. Please place wiki-18.jsonl.gz in {wiki_dir}")
        return None

    print("Extracting tar archive from gzipped file...")
    try:
        with tarfile.open(wiki_gz_file, "r:gz") as tar:
            members = tar.getmembers()
            print(f"Found {len(members)} files in archive:")

            json_member = None
            for member in members:
                print(f"  - {member.name} ({member.size} bytes)")
                if member.name.endswith(".jsonl") or member.name.endswith(".json"):
                    json_member = member
                    break

            if json_member is None:
                print("No .jsonl or .json file found in archive!")
                return None

            print(f"Extracting {json_member.name}...")
            with tar.extractfile(json_member) as f_in:
                with open(wiki_file, "wb") as f_out:
                    chunk_size = 8192
                    total_size = json_member.size
                    processed = 0
                    for chunk in tqdm(
                        iter(lambda: f_in.read(chunk_size), b""),
                        desc="Extracting corpus",
                        total=total_size // chunk_size,
                        unit="chunks",
                    ):
                        f_out.write(chunk)
                        processed += len(chunk)
                        if processed % (2000 * 1024 * 1024) == 0:  # print every 2GB
                            tqdm.write(f"Processed {processed / 1024 / 1024:.1f} MB")
    except tarfile.TarError as e:
        print(f"Failed to extract as tar archive: {e}")
        print("Falling back to binary extraction...")
        with gzip.open(wiki_gz_file, "rb") as f_in:
            with open(wiki_file, "wb") as f_out:
                chunk_size = 8192
                for chunk in tqdm(iter(lambda: f_in.read(chunk_size), b""), desc="Extracting corpus"):
                    f_out.write(chunk)

    print(f"Wikipedia corpus extracted to {wiki_file}")
    if remove_gz:
        os.remove(wiki_gz_file)
        print("Removed compressed file to save space")
    return wiki_file


def verify_prebuilt_indices(indices_dir: str):
    part_aa = os.path.join(indices_dir, "part_aa")
    part_ab = os.path.join(indices_dir, "part_ab")

    missing = [p for p in [part_aa, part_ab] if not os.path.exists(p)]
    if missing:
        print("Missing prebuilt index files:")
        for p in missing:
            print(f"  - {p}")
        return None

    print(f"Pre-built indices found in {indices_dir}")
    print("Note: You'll need to concatenate part_aa and part_ab to create the full index")
    print("Run: cat part_aa part_ab > e5_Flat.index")
    return indices_dir


def resolve_wiki_dir(data_dir: str):
    primary = os.path.join(data_dir, "wikipedia")
    legacy = os.path.join(data_dir, "wiki-18-corpus")
    if os.path.exists(primary):
        return primary
    if os.path.exists(legacy):
        return legacy
    return primary


def resolve_indices_dir(data_dir: str):
    primary = os.path.join(data_dir, "prebuilt_indices")
    legacy = os.path.join(data_dir, "wiki-18-e5-index")
    if os.path.exists(primary):
        return primary
    if os.path.exists(legacy):
        return legacy
    return primary


def setup_search_data(data_dir: str = "./search_data", remove_gz: bool = True):
    print(f"Setting up Search data (dense-only) in {data_dir}")
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    wiki_dir = resolve_wiki_dir(data_dir)
    indices_dir = resolve_indices_dir(data_dir)
    Path(wiki_dir).mkdir(parents=True, exist_ok=True)
    Path(indices_dir).mkdir(parents=True, exist_ok=True)

    wiki_file = extract_wikipedia_corpus(wiki_dir, remove_gz=remove_gz)
    prebuilt_dir = verify_prebuilt_indices(indices_dir)

    summary = {
        "wikipedia_corpus": wiki_file,
        "prebuilt_dense_indices": prebuilt_dir,
        "setup_complete": wiki_file is not None and prebuilt_dir is not None,
    }

    summary_file = os.path.join(data_dir, "data_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nData setup {'completed' if summary['setup_complete'] else 'partially completed'}!")
    print(f"Summary saved to {summary_file}")

    if summary["setup_complete"]:
        print("\nNext steps:")
        print("1. Launch dense retrieval server:")
        print("   cd examples/search && bash retrieval/launch_server.sh ./search_data/prebuilt_indices 8000")
        print("2. Start training:")
        print("   cd examples/search && python train_search_agent.py")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Process manually downloaded Search training data & dense indices"
    )
    parser.add_argument("--data_dir", default="./search_data", help="Directory containing data")
    parser.add_argument(
        "--keep_gz",
        action="store_true",
        help="Keep wiki-18.jsonl.gz after extraction",
    )

    args = parser.parse_args()
    setup_search_data(
        data_dir=args.data_dir,
        remove_gz=not args.keep_gz,
    )


if __name__ == "__main__":
    main()
