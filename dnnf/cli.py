import argparse

from collections import defaultdict
from pathlib import Path

from . import __version__
from . import backends


class LongHelp(argparse.Action):
    def __init__(
        self,
        option_strings,
        dest=argparse.SUPPRESS,
        default=argparse.SUPPRESS,
        help=None,
    ):
        super().__init__(
            option_strings=option_strings,
            dest=dest,
            default=default,
            nargs=0,
            help=help,
        )

    def __call__(self, parser, namespace, values, option_string=None):
        parser.print_help()
        parser.exit()

    def __call__(self, parser, namespace, values, option_string=None):
        parser.print_help()
        available_backends = "\n  ".join(
            backends.get_backend_choices(group_equivalent=True)
        )
        available_backends_str = f"\navailable backends:\n  {available_backends}"
        print(available_backends_str)
        parser.exit()


class HelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter
):
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


def float_type(parser, name, x):
    if len(x) > 1:
        raise parser.error(f"Too many values for parameter {name}.")
    return float(x[0])


def int_type(parser, name, x):
    if len(x) > 1:
        raise parser.error(f"Too many values for parameter {name}.")
    return int(x[0])


def str_type(parser, name, x):
    if len(x) > 1:
        raise parser.error(f"Too many values for parameter {name}.")
    return str(x[0])


def bool_type(parser, name, x):
    if len(x) > 1:
        raise parser.error(f"Too many values for parameter {name}.")
    if len(x) == 1:
        return x[0].lower() in ["true", "1"]
    return True


def literal_type(parser, name, x):
    import ast

    if len(x) > 1:
        raise parser.error(f"Too many values for parameter {name}.")
    try:
        return ast.literal_eval(x[0])
    except:
        return x[0]


def array_type(parser, name, x):
    import ast
    import numpy as np

    if len(x) > 1:
        raise parser.error(f"Too many values for parameter {name}.")
    lst = ast.literal_eval(x[0])
    if not isinstance(lst, list):
        raise parser.error(f"Parameter {name} requires a list type value.")
    return np.asarray(lst)


class SetParameter(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings=option_strings, dest=dest, **kwargs)
        self.parameters = defaultdict(lambda: {})
        self.parameter_type = defaultdict(lambda: defaultdict(lambda: literal_type))
        # distillation parameters
        self.parameter_type["pgd"]["alpha"] = float_type

    def __call__(self, parser, namespace, values, option_string=None):
        if len(values) < 2:
            raise argparse.ArgumentTypeError("Too few arguments to --set option")
        method, name, *val = values
        if name in self.parameters:
            raise parser.error(f"Multiple values specified for parameter {name}")
        value = self.parameter_type[method][name](parser, name, val)
        self.parameters[method][name] = value
        items = (getattr(namespace, self.dest) or defaultdict(lambda: {})).copy()
        items[method][name] = value
        setattr(namespace, self.dest, items)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="dnnf",
        description="dnnf - deep neural network falsification",
        formatter_class=HelpFormatter,
    )
    parser.add_argument(
        "--long-help",
        action=LongHelp,
        help="show a longer help message with available falsifier backends and exit",
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
    parser.add_argument(
        "-N",
        "--network",
        action=AddNetwork,
        nargs=2,
        dest="networks",
        metavar=("NAME", "PATH"),
    )

    parser.add_argument(
        "--save-violation",
        metavar="PATH",
        type=Path,
        default=None,
        help="the path to save a found violation",
    )

    prop_format_group = parser.add_mutually_exclusive_group()
    prop_format_group.add_argument(
        "--vnnlib",
        action="store_const",
        const="vnnlib",
        dest="prop_format",
        help="use the vnnlib property format",
    )

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
        "--n_starts",
        default=-1,
        type=int,
        dest="n_starts",
        help="The default number of random starts per sub-property (can be set per backend with --set)",
    )

    parser.add_argument("--cuda", action="store_true", help="use cuda")

    parser.add_argument(
        "--backend",
        type=str,
        nargs="+",
        default=["pgd"],
        help="the falsification backends to use",
        choices=backends.get_backend_choices(),
        metavar="BACKEND",
    )
    parser.add_argument(
        "--set",
        nargs=3,
        dest="parameters",
        action=SetParameter,
        metavar=("BACKEND", "PARAM", "VALUE"),
        help="set parameters for the falsification backend",
    )

    known_args, extra_args = parser.parse_known_args()
    return known_args, extra_args
