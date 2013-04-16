"""
Parse ssearch36 -m10 output and print specified contents
"""

import logging
import sys
import pprint
import csv
from itertools import islice, ifilter, imap, chain

from bioy_pkg.sequtils import homodecodealignment, parse_ssearch36, from_ascii
from bioy_pkg.utils import Opener, Csv2Dict

log = logging.getLogger(__name__)

def build_parser(parser):
    parser.add_argument('alignments',
        default = sys.stdin,
        type = Opener('r'),
        nargs = '?',
        help = 'ssearch -m 10 formatted file')
    parser.add_argument('-o', '--out',
        default = sys.stdout,
        type = Opener('w'),
        help = '(default csv)-formatted output')
    parser.add_argument('-p', '--print-one',
        default = False, action = 'store_true',
        help = 'pretty print first alignment and exit')
    parser.add_argument('-f', '--fieldnames',
        help = 'comma-delimited list of field names to include in output')
    parser.add_argument('--limit',
        type = int,
        metavar = 'N',
        help = 'Print no more than N alignments')
    parser.add_argument('--no-header',
        dest='header',
        action = 'store_false')
    parser.add_argument('-d', '--decode',
        type = Csv2Dict('name', 'rle'),
        default = [],
        nargs = '+',
        help = 'Decode alignment')
    parser.add_argument('--min-zscore',
        default = 0,
        type = float,
        metavar = 'X',
        help = 'Exclude alignments with z-score < X')

def get_fieldnames(aligns):
    top = next(aligns)
    fieldnames = top.keys()
    aligns = chain([top], aligns)
    return fieldnames, aligns

def action(args):
    def decode(aligns):
        aligns['t_seq'], aligns['q_seq'] = homodecodealignment(
                aligns['t_seq'], from_ascii(args.decode[aligns['t_name']]),
                aligns['q_seq'], from_ascii(args.decode[aligns['q_name']]))
        return aligns

    aligns = islice(parse_ssearch36(args.alignments, False), args.limit)
    aligns = ifilter(lambda a: a >= args.min_zscore, aligns)

    if args.decode:
        aligns = imap(lambda a: decode(a), aligns)

    if args.print_one:
        pprint.pprint(aligns.next())
        sys.exit()

    if args.fieldnames:
        fieldnames = args.fieldnames
    else:
        fieldnames, aligns = get_fieldnames(aligns)

    writer = csv.DictWriter(args.out,
            extrasaction = 'ignore',
            fieldnames = fieldnames)

    if args.header:
        writer.writeheader()

    writer.writerows(aligns)
