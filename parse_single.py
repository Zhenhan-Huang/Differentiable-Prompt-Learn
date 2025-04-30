import re
import numpy as np
import os.path as osp
import argparse
from collections import OrderedDict, defaultdict

from dassl.utils import check_isfile, listdir_nohidden


def compute_ci95(res):
    return 1.96 * np.std(res) / np.sqrt(len(res))


def main(args, end_signal):
    metric = {
        "name": args.keyword,
        "regex": re.compile(fr"\* {args.keyword}: ([\.\deE+-]+)%"),
    }
    outputs = []
    for seed in range(1, args.nseeds+1):
        output = OrderedDict()
        good_to_go = False
        fpath = f'{args.directory}/seed{seed}_{args.dataset}_3/log.txt'
        assert osp.isfile(fpath), (
            f'Could not find the file "{fpath}"'
        )

        with open(fpath, "r") as f:
            lines = f.readlines()

            for line in lines:
                line = line.strip()

                if line == end_signal:
                    good_to_go = True

                match = metric["regex"].search(line)

                if match and good_to_go:
                    if "file" not in output:
                        output["file"] = fpath
                    num = float(match.group(1))
                    name = metric["name"]
                    output[name] = num

        if output:
            outputs.append(output)

        assert len(outputs) > 0, f"Nothing found in {fpath}"

    metrics_results = defaultdict(list)

    for output in outputs:
        msg = ""
        for key, value in output.items():
            if isinstance(value, float):
                msg += f"{key}: {value:.2f}%. "
            else:
                msg += f"{key}: {value}. "
            if key != "file":
                metrics_results[key].append(value)
        print(msg)

    output_results = OrderedDict()

    print("===")
    print(f"Summary of directory: {args.directory}")
    for key, values in metrics_results.items():
        avg = np.mean(values)
        std = compute_ci95(values) if args.ci95 else np.std(values)
        print(f"* {key}: {avg:.2f}% +- {std:.2f}%")
        output_results[key] = avg
    print("===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=str, help="path to directory")
    parser.add_argument(
        "--nseeds", type=int, help="parse multiple experiments"
    )
    parser.add_argument("--dataset", type=str)
    parser.add_argument(
        "--ci95", action="store_true", help=r"compute 95\% confidence interval"
    )
    parser.add_argument("--test-log", action="store_true", help="parse test-only logs")
    parser.add_argument(
        "--keyword", default="accuracy", type=str, help="which keyword to extract"
    )
    args = parser.parse_args()

    end_signal = "Finish training"
    if args.test_log:
        end_signal = "=> result"

    main(args, end_signal)
