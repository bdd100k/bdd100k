"""Run data commands in parallel."""

from __future__ import annotations

import argparse
import os
from os.path import dirname, join, splitext
from subprocess import DEVNULL, check_call
from typing import Callable, List

from joblib import Parallel, delayed
from tqdm import tqdm


def parse_arguments() -> argparse.Namespace:
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="Prepare data")
    parser.add_argument(
        "cmd", type=str, choices=["copy_web", "copy", "zip", "unzip"]
    )
    parser.add_argument("--input-dir", "-i", type=str, required=True)
    parser.add_argument("--output-dir", "-o", type=str, required=True)
    parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=1,
        help="Process multiple videos in parallel.",
    )

    args = parser.parse_args()
    return args


def copy(src: str, target: str) -> None:
    """Copy from src to target."""
    out_dir = dirname(target)
    os.makedirs(out_dir, exist_ok=True)
    cmd = ["rsync", "-aq", src, out_dir]
    pipe = DEVNULL
    check_call(cmd, stdout=pipe, stderr=pipe)


def zipdir(src: str, target: str) -> None:
    """Zip the src folder."""
    out_dir = dirname(target)
    os.makedirs(out_dir, exist_ok=True)
    cmd = ["zip", "-r", "-j", "-0", "-q", target + ".zip", src]
    pipe = DEVNULL
    check_call(cmd, stdout=pipe, stderr=pipe)


def unzip(src: str, target: str) -> None:
    """Unzip the src folder."""
    out_dir = dirname(target)
    os.makedirs(out_dir, exist_ok=True)
    cmd = ["unzip", "-d", splitext(target)[0], src]
    pipe = DEVNULL
    check_call(cmd, stdout=pipe, stderr=pipe)


def create_subpath(filepath: str) -> str:
    """Create subfolder path."""
    return join("/".join(filepath[:3]), filepath)


def listdir(in_dir: str) -> List[str]:
    """List items in the directory."""
    return sorted(os.listdir(in_dir))


def run() -> None:
    """Run the command."""
    args = parse_arguments()
    in_dir = args.input_dir
    out_dir = args.output_dir
    jobs = args.jobs
    cmd = args.cmd
    items = listdir(in_dir)
    out_paths = [join(out_dir, p) for p in items]
    in_paths = [join(in_dir, p) for p in items]

    if cmd == "copy_web":
        out_paths = [join(out_dir, create_subpath(p)) for p in items]

    func: Callable[[str, str], None]
    if cmd in ("copy_web", "copy"):
        func = copy
    elif cmd == "zip":
        func = zipdir
    elif cmd == "unzip":
        func = unzip

    Parallel(n_jobs=jobs, backend="multiprocessing")(
        delayed(func)(in_paths[i], out_paths[i])
        for i in tqdm(range(len(items)))
    )


if __name__ == "__main__":
    run()
