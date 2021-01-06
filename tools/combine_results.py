#!/usr/bin/env python
import argparse
import pandas as pd

from functools import partial
from pathlib import Path


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--results",
        dest="results_path",
        type=Path,
        default=Path("results"),
        help="path to directory with results files",
    )
    return parser.parse_args()


def load_results(path):
    if path.is_dir():
        df = pd.DataFrame(
            [],
            columns=(
                "Artifact",
                "Variant",
                "Method",
                "ProblemId",
                "Result",
                "TotalTime",
            ),
        )
        for results_file_path in path.iterdir():
            print(results_file_path)
            run_config = [
                x
                for x in results_file_path.stem.split(".")
                if x not in ["falsify", "verify"]
            ]
            artifact = run_config[0]
            variant = "None"
            i = 1
            if artifact == "neurifydave":
                i = 2
                artifact_name = "Neurify-DAVE"
                variant = f"{run_config[1]}"
            elif artifact == "acas":
                artifact_name = "ACAS Xu"
            elif artifact == "ghpr":
                artifact_name = "GHPR"
            elif artifact == "diff":
                artifact_name = "Differencing"
                if run_config[1] == "global":
                    variant = "global"
                    i = 2
                else:
                    variant = ".".join(run_config[1:3])
                    i = 3
            method = ".".join(run_config[i:]).split("_", maxsplit=1)[-1]

            df_ = pd.read_csv(results_file_path)
            df_["Artifact"] = artifact_name
            df_["Variant"] = variant
            df_["Method"] = method
            if "FalsificationTime" in df.columns:
                df = df.drop("FalsificationTime", axis=1)

            if artifact == "ghpr":
                df_["Variant"] = df_["ProblemId"].str.split("_", expand=True)[0]

            df = pd.concat([df, df_.dropna(subset=["Result"])])
        df.loc[df["Result"].str.contains("VerificationRunnerError"), "Result"] = "!"
        df.loc[df["Result"].str.contains("Error"), "Result"] = "error"

        def combine_results(df, method_name="falsification"):
            df_ = pd.DataFrame(
                [[method_name, None, None]],
                columns=("Method", "Result", "TotalTime"),
            )
            results = df["Result"].unique()
            if "sat" in results:
                assert "unsat" not in results
                df_["Result"] = "sat"
                df_["TotalTime"] = df.loc[df["Result"] == "sat", "TotalTime"].min()
            elif "unsat" in results:
                assert "sat" not in results
                df_["Result"] = "unsat"
                df_["TotalTime"] = df.loc[df["Result"] == "unsat", "TotalTime"].min()
            elif len(results) == 1:
                df_["Result"] = list(results)[0]
                df_["TotalTime"] = df["TotalTime"].max()
            else:
                df_["Result"] = "unknown"
                df_["TotalTime"] = df["TotalTime"].max()
            return df_

        verifiers = ["eran", "neurify", "reluplex", "planet"]
        df_F = df[~df["Method"].isin(verifiers)].groupby(
            ["Artifact", "Variant", "ProblemId"]
        )
        df = pd.concat(
            [
                df,
                df_F.apply(
                    partial(combine_results, method_name="falsification")
                ).reset_index(),
            ]
        )

        df_V = df[df["Method"].isin(verifiers)].groupby(
            ["Artifact", "Variant", "ProblemId"]
        )
        df = pd.concat(
            [
                df,
                df_V.apply(
                    partial(combine_results, method_name="verification")
                ).reset_index(),
            ]
        )

        df_G = df[df["Method"].isin(["falsification", "verification"])].groupby(
            ["Artifact", "Variant", "ProblemId"]
        )
        df = pd.concat(
            [
                df,
                df_G.apply(
                    partial(combine_results, method_name="global")
                ).reset_index(),
            ]
        )

        for col_name in df.columns:
            if col_name.startswith("level"):
                df = df.drop(col_name, axis=1)
        if "index" in df:
            df = df.drop("index", axis=1)

        df.loc[df["Result"] == "sat", "Result"] = "falsified"
        df.loc[df["Result"] == "unsat", "Result"] = "verified"
        df.to_csv(f"{path}.csv", index=False)
    else:
        df = pd.read_csv(str(path))
    return df


def main(args):
    df = load_results(args.results_path)



if __name__ == "__main__":
    main(_parse_args())
