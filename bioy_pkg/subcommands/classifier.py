# This file is part of Bioy
#
#    Bioy is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Bioy is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Bioy.  If not, see <http://www.gnu.org/licenses/>.

"""Classify sequences by grouping blast output by matching taxonomic names

Optional grouping by specimen and query sequences

Running the program
-------------------

::

    positional arguments:
      blast_file            CSV tabular blast file of query and subject hits.
      seq_info              File mapping reference seq name to tax_id
      taxonomy              Table defining the taxonomy for each tax_id

    optional arguments:
      -h, --help            show this help message and exit
      --threads NUM         Number of threads (CPUs). Can also specify with
                            environment variable THREADS_ALLOC. [32]
      --copy-numbers CSV    Estimated 16s rRNA gene copy number for each
                            tax_ids (CSV file with columns: tax_id, median)
      --rank-thresholds CSV
                            Columns [tax_id,ranks...]
      --specimen-map CSV    CSV file with columns (name, specimen) assigning
                            sequences to groups. The default behavior is to
                            treat all query sequences as
                            belonging to one specimen.
      -w CSV, --weights CSV
                            Optional headless csv file with columns 'seqname',
                            'count' providing weights for each query sequence
                            described in the blast input (used, for example, to
                            describe cluster sizes for corresponding cluster
                            centroids).
      -o FILE, --out FILE   Classification results.
      -O FILE, --details-out FILE
                            Optional details of taxonomic assignments.
      --details-full        do not limit out_details to only larget cluster per
                            assignment
      --group-def INTEGER   define a group threshold for a particular rank
                            overriding --max-group-size. example:
                            genus:2 (NOT IMPLEMENTED)
      --has-header          specify this if blast data has a header
      --min-identity PERCENT
                            minimum identity threshold for accepting matches
                            [>= 0.0]
      --max-identity PERCENT
                            maximum identity threshold for accepting matches
                            [<= 100.0]
      --min-cluster-size INTEGER
                            minimum cluster size to include in classification
                            output [1]
      --min-coverage PERCENT
                            percent of alignment coverage of blast result [0.0]
      --specimen LABEL      Single group label for reads
      --starred PERCENT     Names of organisms for which at least one reference
                            sequence has pairwise identity with a query
                            sequence of at least PERCENT will be marked with an
                            asterisk[100.0]
      --max-group-size INTEGER
                            group multiple target-rank assignments that excede
                            a threshold to a higher rank [3]

Positional arguments
++++++++++++++++++++

blast_file
==========

A csv file with columns **qseqid**, **sseqid**, **pident**,
**qstart**, **qend**, **qlen** and **qcovs**.

.. note:: The actual header is optional if using default blast out format but
          if present make sure to use the --has-header switch

seq_info
========

A csv file with minimum columns **seqname** and **tax_id**.  Additional
columns will be included in the details output.

taxonomy
========

A csv file with columns **tax_id**, **rank** and **tax_name**, plus at least
one additional rank column(s) creating a taxonomic tree such as **species**,
**genus**, **family**, **class**, **pylum**, **kingdom** and/or **root**.
The rank columns also give an order of specificity from right to left,
least specific to most specific respectively.

Optional input
++++++++++++++

rank-thresholds
===============

TODO

copy-numbers
============

Below is an *example* copy numbers csv with the required columns:

    ====== ==================== ======
    tax_id tax_name             median
    ====== ==================== ======
    155977 Acaryochloris        2.00
    155978 Acaryochloris marina 2.00
    434    Acetobacter          5.00
    433    Acetobacteraceae     3.60
    ====== ==================== ======

weights
=======

Headerless file containing two columns specifying the seqname (clustername) and
weight (or number of sequences in the cluster).

Output
++++++

out
===

A csv with columns and headers as in the example below:

    =========== =============== ======================================
     specimen    assignment_id   assignment
    =========== =============== ======================================
      039_3      0               Pseudomonas mendocina;Pseudonocardia
      039_3      1               Rhizobiales
      039_3      2               Alcaligenes faecalis*
      039_3      3               [no blast result]
    =========== =============== ======================================

    ======= ============= =============
     low     max_percent   min_percent
    ======= ============= =============
     95.00   99.02         95.74
     95.00   98.91         95.31
     99.00   100.00        99.00

    ======= ============= =============

    ================= ======= =========== ===========
     condensed_rank   reads   pct_reads   clusters
    ================= ======= =========== ===========
     species          6       35.29       1
     genus            5       29.41       1
     species          5       29.41       1
                      1       5.88        1
    ================= ======= =========== ===========

details-out
===========

A csv that is basically a blast results breakdown of the `out`_ output.

Internal functions
------------------

Known bugs
----------

Tax_ids of valid Blast hits (hits that meet their rank thresholds) may be
assigned tax_ids of a higher threshold that *could* represent invalid tax_ids
(tax_ids that may *not* have passed the rank threshold).
"""

import os
import sys
import logging

import pandas as pd
import math

from bioy_pkg import sequtils, _data as datadir

log = logging.getLogger(__name__)

ASSIGNMENT_TAX_ID = 'assignment_tax_id'


def raw_filtering(blast_results, min_coverage=None,
                  max_identity=None, min_identity=None):
    """run raw hi, low and coverage filters and output log information
    """

    blast_results_len = len(blast_results)

    if min_coverage:
        # run raw hi, low and coverage filters
        blast_results = blast_results[
            blast_results['qcovs'] >= min_coverage]

        blast_results_post_len = len(blast_results)

        len_diff = blast_results_len - blast_results_post_len
        if len_diff:
            log.warn('dropping {} sequences below '
                     'coverage threshold'.format(len_diff))

        blast_results_len = blast_results_post_len

    if max_identity:
        blast_results = blast_results[
            blast_results['pident'] <= max_identity]

        blast_results_post_len = len(blast_results)

        len_diff = blast_results_len - blast_results_post_len
        if len_diff:
            log.warn('dropping {} sequences above max_identity'.format(
                len_diff))

        blast_results_len = blast_results_post_len

    if min_identity:
        blast_results = blast_results[
            blast_results['pident'] >= min_identity]

        blast_results_post_len = len(blast_results)

        len_diff = blast_results_len - blast_results_post_len
        if len_diff:
            log.warn('dropping {} sequences below min_identity'.format(
                len_diff))

        blast_results_len = blast_results_post_len

    return blast_results


def round_up(x):
    """round up any x < 0.01
    """
    return max(0.01, x)


def star(df, starred):
    """Assign boolean if any items in the
    dataframe are above the star threshold.
    """

    df['starred'] = df.pident.apply(lambda x: x >= starred).any()
    return df


def condense_ids(
        df, tax_dict, ranks, max_group_size, threshold_assignments=False):
    """
    Create mapping from tax_id to its condensed id.  Also creates the
    assignment hash on either the condensed_id or assignment_tax_id decided
    by the --split-condensed-assignments switch.

    By taking a hash of the set (frozenset) of ids the qseqid is given a
    unique identifier (the hash).  Later, we will use this hash and
    assign an ssignment name based on either the set of condensed_ids or
    assignment_tax_ids.  The modivation for using a hash rather than
    the actual assignment text for grouping is that the assignment text
    can contains extra annotations that are independent of which
    assignment group a qseqid belongs to such as a 100% id star.
    """

    condensed = sequtils.condense_ids(
        df[ASSIGNMENT_TAX_ID].unique(),
        tax_dict,
        ranks=ranks,
        max_size=max_group_size)

    condensed = pd.DataFrame(
        condensed.items(),
        columns=[ASSIGNMENT_TAX_ID, 'condensed_id'])
    condensed = condensed.set_index(ASSIGNMENT_TAX_ID)

    if threshold_assignments:
        assignment_hash = hash(frozenset(condensed.index.unique()))
    else:
        assignment_hash = hash(frozenset(condensed['condensed_id'].unique()))

    condensed['assignment_hash'] = assignment_hash
    return df.join(condensed, on=ASSIGNMENT_TAX_ID)


def assign(df, tax_dict):
    """Create str assignment based on tax_ids str and starred boolean.
    """

    ids_stars = df.groupby(by=['condensed_id', 'starred']).groups.keys()
    df['assignment'] = sequtils.compound_assignment(ids_stars, tax_dict)
    return df


def assignment_id(df):
    """Resets and drops the current dataframe's
    index and sets it to the assignment_hash

    assignment_id is treated as a string identifier to account
    for hits in details with no assignment or assignment_id
    """
    df = df.reset_index(drop=True)  # specimen is retained in the group key
    df.index.name = 'assignment_id'
    df.index = df.index.astype(str)
    return df


def best_rank(s, ranks):
    """Create aggregate columns for assignments.

    `ranks' are sorted with less specific first for example:

    ['root', 'kingdom', 'phylum', 'order', 'family',
     'genus', 'species_group', 'species']

    so when sorting the indexes the most specific
    rank will be in the iloc[-1] location
    """

    value_counts = s.value_counts()
    if len(value_counts) == 0:
        # [no blast result]
        return None
    else:
        def specificity(r):
            return ranks.index(r) if r in ranks else -1
        majority_rank = value_counts[value_counts == value_counts.max()]
        majority_rank.name = 'rank'
        majority_rank = majority_rank.to_frame()
        majority_rank['specificity'] = majority_rank['rank'].apply(specificity)
        majority_rank = majority_rank.sort_values(by=['specificity'])
        # most precise rank is at the highest index (-1)
        return majority_rank.iloc[-1].name


def find_tax_id(series, valids, r, ranks):
    """Return the most taxonomic specific tax_id available for the given
    Series.  If a tax_id is already present in valids[r] then return None.
    """

    index = ranks.index(r)
    series = series[ranks[index:]]
    series = series[~series.isnull()]
    found = series.head(n=1)
    key = found.index.values[0]
    value = found.values[0]
    return value if value not in valids[key].unique() else None


def select_valid_hits(df, ranks):
    """Return valid hits of the most specific rank that passed their
    corresponding rank thresholds.  Hits that pass their rank thresholds
    but do not have a tax_id at that rank will be bumped to a less specific
    rank id and varified as a unique tax_id.
    """

    for r in ranks:
        thresholds = df['{}_threshold'.format(r)]
        pidents = df['pident']
        valid = df[thresholds < pidents]

        if valid.empty:
            # Move to higher rank
            continue

        tax_ids = valid[r]
        na_ids = tax_ids.isnull()

        # Occasionally tax_ids will be missing at a certain rank.
        # If so use the next less specific tax_id available
        if na_ids.all():
            continue

        if na_ids.any():
            # bump up missing taxids
            have_ids = valid[tax_ids.notnull()]
            found_ids = valid[na_ids].apply(
                find_tax_id, args=(have_ids, r, ranks), axis=1)
            tax_ids = have_ids[r].append(found_ids)

        valid[ASSIGNMENT_TAX_ID] = tax_ids
        valid['assignment_threshold'] = thresholds
        # return notnull() assignment_threshold valid values
        return valid[valid[ASSIGNMENT_TAX_ID].notnull()]

    # nothing passed
    df[ASSIGNMENT_TAX_ID] = None
    df['assignment_threshold'] = None

    return pd.DataFrame(columns=df.columns)


def calculate_pct_references(df, pct_reference):
    reference_count = df[['tax_id']].drop_duplicates()
    reference_count = reference_count.join(pct_reference, on='tax_id')
    reference_count = reference_count['count'].sum()
    sseqid_count = float(len(df['sseqid'].drop_duplicates()))
    df['pct_reference'] = sseqid_count / reference_count
    return df


def pct(s):
    """Calculate series pct something
    """

    return s / s.sum() * 100


def load_rank_thresholds(
        path=os.path.join(datadir, 'rank_thresholds.csv'), usecols=None):
    """Load a rank-thresholds file.  If no argument is specified the default
    rank_threshold_defaults.csv file will be loaded.
    """

    return pd.read_csv(
        path,
        comment='#',
        usecols=['tax_id'] + usecols,
        dtype=dict(tax_id=str)).set_index('tax_id')


def copy_corrections(copy_numbers, blast_results, user_file=None):
    copy_numbers = pd.read_csv(
        copy_numbers,
        dtype=dict(tax_id=str, median=float),
        usecols=['tax_id', 'median']).set_index('tax_id')

    # get root out (taxid: 1) and set it as the default correction value

    # set index nana (no blast result) to the defaul value
    default = copy_numbers.get_value('1', 'median')
    default_entry = pd.DataFrame(default, index=[None], columns=['median'])
    copy_numbers = copy_numbers.append(default_entry)

    # do our copy number correction math
    corrections = blast_results[
        [ASSIGNMENT_TAX_ID, 'specimen', 'assignment_hash']]
    corrections = corrections.drop_duplicates()
    corrections = corrections.set_index(ASSIGNMENT_TAX_ID)
    corrections = corrections.join(copy_numbers)
    # any tax_id not present will receive default tax_id
    corrections['median'] = corrections['median'].fillna(default)
    corrections = corrections.groupby(
        by=['specimen', 'assignment_hash'], sort=False)
    corrections = corrections['median'].mean()
    return corrections


def join_thresholds(df, thresholds, ranks):
    """Thresholds are matched to thresholds by rank id.

    If a rank id is not present in the thresholds then the next specific
    rank id is used all the way up to `root'.  If the root id still does
    not match then a warning is issued with the taxname and the blast hit
    is dropped.
    """
    with_thresholds = pd.DataFrame(columns=df.columns)  # temp DFrame

    for r in ranks:
        with_thresholds = with_thresholds.append(
            df.join(thresholds, on=r, how='inner'))
        no_threshold = df[~df.index.isin(with_thresholds.index)]
        df = no_threshold

    # issue warning messages for everything that did not join
    if len(df) > 0:
        tax_names = df['tax_name'].drop_duplicates()
        msg = ('dropping blast hit `{}\', no valid '
               'taxonomic threshold information found')
        tax_names.apply(lambda x: log.warn(msg.format(x)))

    return with_thresholds


def get_compression(io):
    if io is sys.stdout:
        compression = None
    else:
        compress_ops = {'.gz': 'gzip', '.bz2': 'bz2'}
        ext = os.path.splitext(io)[-1]
        compression = compress_ops.get(ext, None)
    return compression


def build_parser(parser):
    # required inputs
    parser.add_argument(
        'blast_file',
        help="""CSV tabular blast file of
                query and subject hits, containing
                at least {}.""".format(sequtils.BLAST_FORMAT_DEFAULT))
    parser.add_argument(
        'seq_info',
        help='File mapping reference seq name to tax_id')
    parser.add_argument(
        'taxonomy',
        help="""Table defining the taxonomy for each tax_id""")

    # optional inputs
    parser.add_argument(
        '--copy-numbers', metavar='CSV',
        help="""Estimated 16s rRNA gene copy number for each tax_ids
        (CSV file with columns: tax_id, median)""")
    parser.add_argument(
        '--rank-thresholds', metavar='CSV',
        help="""Columns [tax_id,ranks...]""")
    parser.add_argument(
        '--specimen-map', metavar='CSV',
        help="""CSV file with columns (name, specimen) assigning sequences to
        groups. The default behavior is to treat all query sequences
        as belonging to one specimen.""")
    parser.add_argument(
        '-w', '--weights', metavar='CSV',
        help="""Optional headless csv file with columns 'seqname',
        'count' providing weights for each query sequence described in
        the blast input (used, for example, to describe cluster sizes
        for corresponding cluster centroids).""")

    # common outputs
    parser.add_argument(
        '-o', '--out',
        default=sys.stdout,
        metavar='FILE',
        help="Classification results.")
    parser.add_argument(
        '-O', '--details-out',
        metavar='FILE',
        help="""Optional details of taxonomic assignments.""")

    # switches and options
    parser.add_argument(
        '--details-full', action='store_true',
        help='do not limit out_details to only larget cluster per assignment')
    parser.add_argument(
        '--hits-below-threshold',
        action='store_true',
        help=('Hits that were below the best-rank threshold '
              'will be included in the details'))
    parser.add_argument(
        '--include-ref-rank',
        action='append',
        default=[],
        help=("Given a single rank (species,genus,etc), "
              "include each reference "
              "sequence's tax_id as $\{rank\}_id and its taxonomic name as "
              "$\{rank\}_name in details output "))
    parser.add_argument(
        '--group-def', metavar='INTEGER', action='append',
        default=[], help="""define a group threshold for a
        particular rank overriding --max-group-size. example:
        genus:2 (NOT IMPLEMENTED)""")
    parser.add_argument(
        '--has-header', action='store_true',
        help='specify this if blast data has a header')
    parser.add_argument(
        '--min-identity', metavar='PERCENT', type=float,
        help="""minimum identity threshold
        for accepting matches""")
    parser.add_argument(
        '--max-identity', metavar='PERCENT', type=float,
        help="""maximum identity threshold for
        accepting matches""")
    parser.add_argument(
        '--min-cluster-size', default=1, metavar='INTEGER', type=int,
        help="""minimum cluster size to include in
        classification output [%(default)s]""")
    parser.add_argument(
        '--min-coverage', type=float, metavar='PERCENT',
        help='percent of alignment coverage of blast result')
    parser.add_argument(
        '--specimen', metavar='LABEL',
        help="""Single group label for reads""")
    parser.add_argument(
        '--starred', default=100.0, metavar='PERCENT', type=float,
        help="""Names of organisms for which at least one reference
        sequence has pairwise identity with a query sequence of at
        least PERCENT will be marked with an asterisk [%(default)s]""")
    parser.add_argument(
        '--max-group-size', metavar='INTEGER', default=3, type=int,
        help="""group multiple target-rank assignments that excede a
        threshold to a higher rank [%(default)s]""")
    parser.add_argument(
        '--pct-reference', action='store_true',
        help="""include column with percent sseqids per assignment_id
        (NOT IMPLEMENTED)""")
    parser.add_argument(
        '--split-condensed-assignments',
        action='store_true',
        dest='threshold_assignments',
        help=('Do not combine common condensed assignments'))
    parser.add_argument(
        '--limit', type=int, help='limit number of blast results')
    parser.add_argument(
        '--best-n-hits', type=int,
        help="""for each query sequence, filter out all but the best N hits,
        based on the number of mismatches. NOTE: If using this option, blast
        results MUST contain "mismatch" column and column headers
        """)


def action(args):
    # for debugging:
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_rows', None)

    # format blast data and add additional available information
    names = None if args.has_header else sequtils.BLAST_HEADER_DEFAULT
    header = 0 if args.has_header else None
    usecols = ['qseqid', 'sseqid', 'pident', 'qcovs']
    dtypes = dict(qseqid=str, sseqid=str, pident=float, coverage=float)

    if args.best_n_hits:
        usecols.append('mismatch')
        dtypes['mismatch'] = int
    log.info(usecols)
    log.info(dtypes)
    log.info('loading blast results')
    blast_results = pd.read_csv(
        args.blast_file,
        dtype=dtypes,
        names=names,
        na_filter=True,  # False is faster
        header=header,
        usecols=usecols,
        nrows=args.limit)

    if blast_results.empty:
        log.info('blast results empty, exiting.')
        return

    # load specimen-map
    if args.specimen_map:
        # if a specimen_map is defined and a qseqid is not included in the map
        # hits to that qseqid will be dropped (inner join)
        spec_map = pd.read_csv(
            args.specimen_map,
            names=['qseqid', 'specimen'],
            usecols=['qseqid', 'specimen'],
            dtype=str)
        spec_map = spec_map.drop_duplicates()
        spec_map = spec_map.set_index('qseqid')
        blast_results = blast_results.join(spec_map, on='qseqid', how='inner')
    elif args.specimen:
        blast_results['specimen'] = args.specimen
    else:
        blast_results['specimen'] = blast_results['qseqid']  # by qseqid

    # get a set of qseqids for identifying [no blast hits] after filtering
    qseqids = blast_results[['specimen', 'qseqid']].drop_duplicates()

    blast_results_len = len(blast_results)

    log.info('successfully loaded {} blast results for {} query '
             'sequences'.format(blast_results_len, len(qseqids)))

    blast_results = raw_filtering(blast_results)

    # remove no blast hits
    # no_blast_results will be added back later but we do not
    # want to confuse these with blast results filter by joins
    log.info('identifying no_blast_hits')
    blast_results = blast_results[blast_results['sseqid'].notnull()]

    # load seq_info as a bridge to the sequence taxonomy.  Additional
    # columns can be specified to be included in the details-out file
    # such as accession number
    seq_info = pd.read_csv(
        args.seq_info,
        usecols=['seqname', 'tax_id', 'accession'],
        dtype=dict(seqname=str, tax_id=str, accession=str))
    seq_info = seq_info.set_index('seqname')
    # rename index to match blast results column name
    seq_info.index.name = 'sseqid'

    # merge blast results with seq_info - do this early so that
    # refseqs not represented in the blast results are discarded in
    # the merge.
    blast_results_len = len(blast_results)
    log.info('joining seq_info file')
    blast_results = blast_results.join(seq_info, on='sseqid', how='inner')
    len_diff = blast_results_len - len(blast_results)
    if len_diff:
        log.warn('{} subject sequences dropped without '
                 'records in seq_info file'.format(len_diff))

    # load the full taxonomy table.  Rank specificity as ordered from
    # left (less specific) to right (more specific)
    taxonomy = pd.read_csv(args.taxonomy, dtype=str).set_index('tax_id')

    # get the a list of rank columns ordered by specificity (see above)
    # NOTE: we are assuming the rank columns
    ranks = taxonomy.columns.tolist()
    ranks = ranks[ranks.index('root'):]

    # now combine just the rank columns to the blast results
    blast_results_len = len(blast_results)
    log.info('joining taxonomy file')
    blast_results = blast_results.join(
        taxonomy[['tax_name', 'rank'] + ranks], on='tax_id', how='inner')
    len_diff = blast_results_len - len(blast_results)
    if len_diff:
        msg = '{} subject sequences dropped without records in taxonomy file'
        log.warn(msg.format(len_diff))

    # load the default rank thresholds
    rank_thresholds = load_rank_thresholds(usecols=ranks)

    # and any additional thresholds specified by the user
    if args.rank_thresholds:
        rank_thresholds = rank_thresholds.append(
            load_rank_thresholds(path=args.rank_thresholds, usecols=ranks))
        # overwrite with user defined tax_id threshold
        rank_thresholds = rank_thresholds.groupby(level=0, sort=False).last()

    rank_thresholds_cols = ['{}_threshold'.format(c) if c in ranks else c
                            for c in rank_thresholds.columns]
    rank_thresholds.columns = rank_thresholds_cols

    log.info('joining thresholds file')
    blast_results = join_thresholds(
        blast_results, rank_thresholds, ranks[::-1])

    # save the blast_results.columns in case groupby drops all columns
    blast_results_columns = blast_results.columns

    # assign assignment tax ids based on pident and thresholds
    log.info('selecting valid hits')
    blast_results_len = float(len(blast_results))

    valid_hits = blast_results.groupby(
        by=['specimen', 'qseqid'], group_keys=False)
    valid_hits = valid_hits.apply(select_valid_hits, ranks[::-1])

    if args.hits_below_threshold:
        """
        Store all the hits to append to blast_results details later
        """
        hits_below_threshold = blast_results[
            ~blast_results.index.isin(valid_hits.index)]

    blast_results = valid_hits

    if blast_results.empty:
        log.info('all blast results filtered, returning [no blast results]')
        assignment_columns = ['assignment_rank', 'assignment_threshold',
                              'assignment_tax_name', 'condensed_id', 'starred',
                              'assignment', 'assignment_hash',
                              'condensed_rank', ASSIGNMENT_TAX_ID]
        assignment_columns += blast_results_columns.tolist()
        blast_results = pd.DataFrame(columns=assignment_columns)
    else:

        blast_results_post_len = len(blast_results)
        log.info('{} ({:.0%}) valid hits selected'.format(
            blast_results_post_len,
            blast_results_post_len / blast_results_len))

        if args.best_n_hits:
            blast_results_len = len(blast_results)

            def filter_mismatches(df, best_n):
                """
                Filter all hits with more mismatches than the Nth best hit
                """

                threshold = df['mismatch'].nsmallest(best_n).iloc[-1]
                return df[df['mismatch'] <= threshold]

            # Filter hits for each query
            blast_results = blast_results.groupby(
                by=['specimen', 'qseqid'],
                group_keys=False).apply(filter_mismatches, args.best_n_hits)

            blast_results_post_len = len(blast_results)
            log.info('{} ({:.0%}) hits remain after filtering '
                     'on mismatches (--best_n_hits)'.format(
                         blast_results_post_len,
                         blast_results_post_len / blast_results_len))

        # drop unneeded tax and threshold columns to free memory
        for c in ranks + rank_thresholds_cols:
            blast_results = blast_results.drop(c, axis=1)

        # join with taxonomy for tax_name and rank
        blast_results = blast_results.join(
            taxonomy[['tax_name', 'rank']],
            rsuffix='_assignment',
            on=ASSIGNMENT_TAX_ID)

        # no join(rprefix) (yet)
        blast_results = blast_results.rename(
            columns={'tax_name_assignment': 'assignment_tax_name',
                     'rank_assignment': 'assignment_rank'})

        # TODO: this is relatively slow, need to integrate
        # pandas into sequtils.condense_ids
        tax_dict = {i: t.to_dict() for i, t in taxonomy.fillna('').iterrows()}

        # create condensed assignment hashes by qseqid
        msg = 'condensing group tax_ids to size {}'.format(args.max_group_size)
        log.info(msg)
        blast_results = blast_results.groupby(
            by=['specimen', 'qseqid'], sort=False, group_keys=False)
        blast_results = blast_results.apply(
            condense_ids,
            tax_dict,
            ranks,
            args.max_group_size,
            threshold_assignments=args.threshold_assignments)

        blast_results = blast_results.join(
            taxonomy[['rank']], on='condensed_id', rsuffix='_condensed')

        blast_results = blast_results.rename(
            columns={'rank_condensed': 'condensed_rank'})

        # star condensed ids if one hit meets star threshold
        by = ['specimen', 'assignment_hash', 'condensed_id']
        blast_results = blast_results.groupby(
            by=by, sort=False, group_keys=False)
        blast_results = blast_results.apply(star, args.starred)

        # assign names to assignment_hashes
        blast_results = blast_results.sort_values(by='assignment_hash')
        log.info('creating compound assignments')
        blast_results = blast_results.groupby(
            by=['specimen', 'assignment_hash'], sort=False, group_keys=False)
        blast_results = blast_results.apply(assign, tax_dict)

        # Foreach ref rank:
        # - merge with taxonomy, extract rank_id, rank_name
        for rank in args.include_ref_rank:
            blast_results[rank + '_id'] = blast_results.merge(
                taxonomy, left_on='tax_id',
                right_index=True,
                how='left')[rank].fillna(0)
            blast_results[rank + '_name'] = blast_results.merge(
                taxonomy,
                left_on=rank + '_id',
                right_index=True,
                how='left')['tax_name_y']

    # merge qseqids that have no hits back into blast_results
    blast_results = blast_results.merge(qseqids, how='outer')

    # assign seqs that had no results to [no blast_result]
    no_hits = blast_results['sseqid'].isnull()
    blast_results.loc[no_hits, 'assignment'] = '[no blast result]'
    blast_results.loc[no_hits, 'assignment_hash'] = 0

    # concludes our blast details, on to output summary
    log.info('summarizing output')

    # index by specimen and assignment_hash and add assignment column
    index = ['specimen', 'assignment_hash']
    output = blast_results[index + ['assignment']].drop_duplicates()
    output = output.set_index(index)

    # assignment level stats
    assignment_stats = blast_results.groupby(by=index, sort=False)
    output['max_percent'] = assignment_stats['pident'].max()
    output['min_percent'] = assignment_stats['pident'].min()
    output['min_threshold'] = assignment_stats['assignment_threshold'].min()
    output['best_rank'] = assignment_stats['condensed_rank'].apply(
        best_rank, ranks)

    # qseqid cluster stats
    weights = blast_results[
        ['qseqid', 'specimen', 'assignment_hash', 'assignment_threshold']]
    weights = weights.drop_duplicates().set_index('qseqid')

    if args.weights:
        weights_file = pd.read_csv(
            args.weights,
            names=['qseqid', 'weight'],
            dtype=dict(qseqid=str, weight=float),
            index_col='qseqid')
        weights = weights.join(weights_file)
        # enforce weight dtype as float and unlisted qseq's to weight of 1.0
        weights['weight'] = weights['weight'].fillna(1.0).astype(float)
    else:
        weights['weight'] = 1.0

    cluster_stats = weights[['specimen', 'assignment_hash', 'weight']]
    cluster_stats = cluster_stats.reset_index().drop_duplicates()
    cluster_stats = cluster_stats.groupby(
        by=['specimen', 'assignment_hash'], sort=False)

    output['reads'] = cluster_stats['weight'].sum()
    output['clusters'] = cluster_stats.size()

    # specimen level stats
    specimen_stats = output.groupby(level='specimen', sort=False)
    output['pct_reads'] = specimen_stats['reads'].apply(pct)

    # copy number corrections
    if args.copy_numbers:
        corrections = copy_corrections(args.copy_numbers, blast_results)
        output['corrected'] = output['reads'] / corrections
        # reset corrected counts to int before calculating pct_corrected
        output['corrected'] = output['corrected'].apply(math.ceil)
        output['corrected'] = output['corrected'].fillna(1).astype(int)
        # create pct_corrected column
        output['pct_corrected'] = specimen_stats['corrected'].apply(pct)
        output['pct_corrected'] = output['pct_corrected'].map(round_up)

    # round reads for output
    output['reads'] = output['reads'].apply(round).astype(int)
    output['pct_reads'] = output['pct_reads'].map(round_up)

    # sort output by:
    # 1) specimen -- Data Frame is already grouped by specimen
    # 2) read/corrected count
    # 3) cluster count
    # 4) alpha assignment
    columns = ['corrected'] if args.copy_numbers else ['reads']
    columns += ['clusters', 'assignment']
    output = output.sort_values(by=columns, ascending=False)
    output = output.reset_index(level='assignment_hash')

    # Sort index (specimen) in preparation for groupby.
    # Use stable sort (mergesort) to preserve sortings (1-4);
    # default algorithm is not stable
    output = output.sort_index(kind='mergesort')

    # one last grouping on the sorted output plus assignment ids by specimen
    output = output.groupby(level="specimen", sort=False).apply(assignment_id)

    # output to details.csv.bz2
    if args.details_out:
        # Annotate details with classification columns
        blast_results = blast_results.merge(output.reset_index(), how='left')

        if not args.details_full:
            """
            by using the assignment_threshold we will get multiple 'largest'
            centroids for --max-group-size combined assignments
            """
            # groupby will drop NA values so we must fill them with 0
            weights['assignment_threshold'] = weights[
                'assignment_threshold'].fillna(0)
            largest = weights.groupby(
                by=['specimen', 'assignment_hash', 'assignment_threshold'],
                sort=False)
            largest = largest.apply(lambda x: x['weight'].nlargest(1))
            largest = largest.reset_index()
            # assignment_threshold will conflict with blast_results NA values
            largest = largest.drop('assignment_threshold', axis=1)
            blast_results = blast_results.merge(largest)

        details_columns = ['specimen', 'assignment_id', 'tax_name', 'rank',
                           'assignment_tax_name', 'assignment_rank', 'pident',
                           'tax_id', ASSIGNMENT_TAX_ID, 'condensed_id',
                           'accession', 'qseqid', 'sseqid', 'starred',
                           'assignment_threshold']
        ref_rank_columns = [rank + '_id' for rank in args.include_ref_rank]
        ref_rank_columns += [rank + '_name' for rank in args.include_ref_rank]
        details_columns += ref_rank_columns

        if args.hits_below_threshold:
            """
            append assignment_thresholds and append to --details-out
            """
            deets_cols = hits_below_threshold.columns
            deets_cols &= set(details_columns)
            hits_below_threshold = hits_below_threshold[list(deets_cols)]
            threshold_cols = ['specimen', 'qseqid', 'assignment_threshold']
            assignment_thresholds = blast_results[threshold_cols]
            assignment_thresholds = assignment_thresholds.drop_duplicates()
            hits_below_threshold = hits_below_threshold.merge(
                assignment_thresholds, how='left')
            blast_results = pd.concat(
                [blast_results, hits_below_threshold], ignore_index=True)

        # sort details for consistency and ease of viewing
        blast_results = blast_results.sort_values(by=details_columns)

        blast_results.to_csv(
            args.details_out,
            compression=get_compression(args.details_out),
            columns=details_columns,
            header=True,
            index=False,
            float_format='%.2f')

    # was required to merge with details above but not needed now
    output = output.drop('assignment_hash', axis=1)

    # output results
    output.to_csv(
        args.out,
        index=True,
        float_format='%.2f',
        compression=get_compression(args.out))
