"""
Parse region between primers from fasta file
"""

import logging
import sys

from subprocess import Popen, PIPE
from csv import DictWriter
from cStringIO import StringIO

from bioy_pkg.sequtils import BLAST_FORMAT
from bioy_pkg.utils import Opener

log = logging.getLogger(__name__)

def build_parser(parser):
    parser.add_argument('fasta',
            default = '-',
            help = 'input fasta file')
    parser.add_argument('-o', '--out',
            type = Opener('w'),
            default = sys.stdout,
            help = 'tabulated BLAST results with the following headers {}'.format(BLAST_FORMAT))
    parser.add_argument('-d', '--database',
            help = 'a blast database')
    parser.add_argument('--limit',
            type = int,
            help = 'maximum number of query sequences to read from the alignment')
    parser.add_argument('--raw',
            action = 'store_true',
            help = 'raw tabular blast output')
    parser.add_argument('--header',
            action = 'store_true',
            help = 'output header')
    parser.add_argument('-u', '--usearch',
            action = 'store_true',
            help = 'usearch against a fasta file')
    parser.add_argument('--strand',
            default = 'plus',
            choices = ['plus', 'minus', 'both'],
            help = """query strand(s) to search against database/subject.
                      default = %(default)s""")
    parser.add_argument('--threads',
            default = '1',
            help = """Number of threads (CPUs) to use in the BLAST search.
                      default = %(default)s""")
    parser.add_argument('--time',
            action = 'store_true',
            help = 'include blast time output')
    parser.add_argument('--log',
            default = sys.stderr,
            type = Opener('w'),
            help = 'log file, default: stderr')
    parser.add_argument('--id',
            default = 0.9,
            type = float,
            help = 'minimum identity for accepted values default [%(default)s]')
    parser.add_argument('--max',
            help = 'maximum number of alignments to keep')

def action(args):
    command = ['usearch6_64']
    command += ['-usearch_global', args.fasta]
    command += ['-threads', args.threads]
    command += ['-id', str(args.id)]
    command += ['-db', args.database]
    command += ['-strand', args.strand]
    command += ['-blast6out', '/dev/stdout']

    if args.max:
        command += ['-maxaccepts', args.max]

    pipe = Popen(command, stdout = PIPE, stderr = PIPE)
    results, errors = pipe.communicate()

    log.error(errors)

    fieldnames = BLAST_FORMAT.split()[1:]

    out = DictWriter(args.out, fieldnames = fieldnames)

    if args.header:
        out.writeheader()

    for r in StringIO(results).readlines()[6:]:
        vals = r.split('\t')
        vals = vals[:3] + [vals[6], vals[7] , '']
        out.writerow(dict(zip(fieldnames, vals)))