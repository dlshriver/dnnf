import argparse

from pathlib import Path

from . import __version__


class HelpFormatter(argparse.HelpFormatter):
    def _format_action_invocation(self, action):
        if not action.option_strings:
            (metavar,) = self._metavar_formatter(action, action.dest)(1)
            return metavar
        else:
            parts = []
            # if the Optional doesn't take a value, format is:
            #    -s, --long
            if action.nargs == 0:
                parts.extend(action.option_strings)

            # if the Optional takes a value, format is:
            #    -s ARGS, --long ARGS
            # change to
            #    -s, --long ARGS
            else:
                default = action.dest.upper()
                args_string = self._format_args(action, default)
                for option_string in action.option_strings:
                    parts.append("%s" % option_string)
                parts[-1] += " %s" % args_string
            return ", ".join(parts)


class AddNetwork(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings=option_strings, dest=dest, **kwargs)
        self.networks = {}

    def __call__(self, parser, namespace, values, option_string=None):
        name, network_path_str = values
        if name in self.networks:
            raise parser.error(f"Multiple paths specified for network {name}")
        network_path = Path(network_path_str)
        self.networks[name] = network_path
        items = (getattr(namespace, self.dest) or {}).copy()
        items[name] = network_path
        setattr(namespace, self.dest, items)


def parse_args():
    parser = argparse.ArgumentParser(
        description="falsify - deep neural network falsification",
        prog="falsify",
        formatter_class=HelpFormatter,
    )
    parser.add_argument("-V", "--version", action="version", version=__version__)
    parser.add_argument("--seed", type=int, default=None, help="the random seed to use")

    verbosity_group = parser.add_mutually_exclusive_group()
    verbosity_group.add_argument("--debug", action="store_true", help=argparse.SUPPRESS)
    verbosity_group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="show messages with finer-grained information",
    )
    verbosity_group.add_argument(
        "-q", "--quiet", action="store_true", help="suppress non-essential messages"
    )

    parser.add_argument("property", type=Path)
    parser.add_argument("-N", "--network", action=AddNetwork, nargs=2, dest="networks")

    parser.add_argument(
        "-p",
        "--processors",
        "--n_proc",
        default=1,
        type=int,
        dest="n_proc",
        help="The maximum number of processors to use",
    )
    parser.add_argument(
        "-S",
        "--starts",
        "--n_starts",
        default=-1,
        type=int,
        dest="n_starts",
        help="The maximum number of random starts per sub-property",
    )

    parser.add_argument("--cuda", action="store_true", help="use cuda")

    parser.add_argument(
        "--tensorfuzz", action="store_true", help="use the tensorfuzz backend"
    )

    known_args, extra_args = parser.parse_known_args()
    return known_args, extra_args
