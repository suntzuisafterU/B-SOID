#!/usr/bin/env python3
"""
Command-line interface for B-SOiD

TODO: Commands to implement:
    - clear logs
    - clear output folder (+ prompt for confirm)
"""

from typing import List
import argparse
import os
import sys

import bsoid

logger = bsoid.config.initialize_logger(__name__)


########################################################################################################################

bsoid_runtime_description = 'BSOID command line utility. Do BSOID stuff. Expand on this later.'

map_command_to_func = {

    'clean': bsoid.app.clear_output_folders,
    'cleanoutput': bsoid.app.clear_output_folders,
    'cleanlogs': bsoid.app.clear_logs,
    'buildandrunlegacy': bsoid.main_LEGACY.test_function_to_build_then_run_py,
    'newbuild': bsoid.app.build_classifier_new_pipeline,
}


########################################################################################################################

# TODO: parse_args() needs to be properly and thoroughly implemented. Until there is enough
#   time to actually do that, use temporary functions below
def parse_args() -> argparse.Namespace:
    """
    Instantiate arguments that will be parsed from the command-line here.
    Regarding HOW these commands will be carried out, implement that elsewhere.
    """
    # Instantiate parser, add arguments as expected on command-line
    parser = argparse.ArgumentParser(description=bsoid_runtime_description)
    parser.add_argument('command')

    # TODO: add more commands, subcommands

    # Parse args, return
    args: argparse.Namespace = parser.parse_args()

    return args


# TODO: do_command() needs to be properly and thoroughly implemented. Until there is enough
#   time to actually do that, use temporary functions below
def do_command(args: argparse.Namespace) -> None:
    # TODO: implement
    return


#### Stand-in functions ################################################################################################

def parse_args_using_sysargv() -> List[str]:
    """
    Stand-in function for parse_args().
    """
    args = sys.argv
    return args


def do_command_from_sysargv_parse(args: List[str]) -> None:
    """
    Stand-in function for do_command(). Because parsing functions using argparse is not complete/ready,
    we use this function for now to execute command-line commands
    """
    cmd = args[1]
    if cmd in map_command_to_func:
        map_command_to_func[cmd]()
    else:
        err = f'Command was not found: `{cmd}` (args: {args}). Check {os.path.abspath(__file__)} to see if application implemented.'
        logger.error(err)
        raise ValueError(err)


### Main execution #####################################################################################################

def main():
    # parse args
    # args = parse_args()
    args = parse_args_using_sysargv()
    # Do stuff
    # do_command(args)
    do_command_from_sysargv_parse(args)
    # print(f'args: {args}')
    # print(f'args.command: {args.command}')

    # End
    pass


if __name__ == '__main__':
    main()
