import argparse
def ParsedArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name",   "-n",  help="Set data save folder name.")
    parser.add_argument("--cores",  "-c",  help="Number of cores to use.")
    parser.add_argument("--show",   "-so", help="Whether to show plots.")
    parser.add_argument("--save",   "-sv", help="Whether to save plots and data.")
    parser.add_argument("--roff",   "-cf", help="Multiplier for realization offset.")
    parser.add_argument("--dispe",  "-de", help="Whether to use default display engine.")
    ParsedPar = parser.parse_args()
    return ParsedPar