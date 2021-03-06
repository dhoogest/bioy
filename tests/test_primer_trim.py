"""
Test subcommands.
"""

import logging

from os import path

from bioy_pkg import main
from bioy_pkg.sequtils import fastalite

from __init__ import TestCaseSuppressOutput, TestBase

log = logging.getLogger(__name__)


class TestPrimerTrimAction(TestCaseSuppressOutput, TestBase):

    def main(self, arguments):
        main(['primer_trim'] + arguments)

    def testHelp(self):
        self.assertRaises(SystemExit, self.main, ['-h'])

    def test01(self):
        out = self.mkoutdir()
        trimmed = path.join(out, 'trimmed.fasta')

        args = [
            self.data('rle_100.fasta'),
            '--left-aligns', self.data('rle_100_left_ssearch.csv.bz2'),
            '--right-aligns', self.data('rle_100_right_ssearch.csv.bz2'),
            '--fasta-out', trimmed
        ]

        self.main(args)

        self.assertTrue(path.exists(trimmed))

        with open(trimmed) as f:
            self.assertEqual(len(list(fastalite(f))), 99)

    def test02(self):
        """
        Include an rle file.
        """

        out = self.mkoutdir()
        trimmed = path.join(out, 'trimmed.fasta')
        trimmed_rle = path.join(out, 'trimmed_rle.csv')

        args = [
            self.data('rle_100.fasta'),
            '--left-aligns', self.data('rle_100_left_ssearch.csv.bz2'),
            '--right-aligns', self.data('rle_100_right_ssearch.csv.bz2'),
            '--fasta-out', trimmed,
            '--rle-out', trimmed_rle
        ]

        # --rle is not provided
        self.assertRaises(SystemExit, self.main, args)

    def test03(self):
        """
        Include an rle file.
        """

        out = self.mkoutdir()
        trimmed = path.join(out, 'trimmed.fasta')
        trimmed_rle = path.join(out, 'trimmed_rle.csv')

        args = [
            self.data('rle_100.fasta'),
            '--left-aligns', self.data('rle_100_left_ssearch.csv.bz2'),
            '--right-aligns', self.data('rle_100_right_ssearch.csv.bz2'),
            '--fasta-out', trimmed,
            '--rle', self.data('rle_100.csv.bz2'),
            '--rle-out', trimmed_rle
        ]

        self.main(args)

        self.assertTrue(path.exists(trimmed))
        self.assertTrue(path.exists(trimmed_rle))

        with open(trimmed) as f:
            self.assertEqual(len(list(fastalite(f))), 99)

        with open(trimmed_rle) as f:
            self.assertEqual(len(f.readlines()), 100)

    def test04(self):
        """
        Include an rle file.
        """

        out = self.mkoutdir()
        trimmed = path.join(out, 'trimmed.fasta')
        trimmed_rle = path.join(out, 'trimmed_rle.csv')

        args = [
            self.data('rle_100.fasta'),
            '--left-aligns', self.data('rle_100_left_ssearch.csv.bz2'),
            '--right-zscore', '80',
            '--right-aligns', self.data('rle_100_right_ssearch.csv.bz2'),
            '--fasta-out', trimmed,
            '--rle', self.data('rle_100.csv.bz2'),
            '--rle-out', trimmed_rle
        ]

        self.main(args)

        with open(trimmed) as f:
            self.assertEqual(len(list(fastalite(f))), 66)

        with open(trimmed_rle) as f:
            self.assertEqual(len(f.readlines()), 67)

    def test05(self):
        """
        Include an rle file.
        """

        out = self.mkoutdir()
        trimmed = path.join(out, 'trimmed.fasta')
        trimmed_rle = path.join(out, 'trimmed_rle.csv')

        args = [
            self.data('rle_100.fasta'),
            '--left-aligns', self.data('rle_100_left_ssearch.csv.bz2'),
            '--right-range', '200,350',
            '--right-aligns', self.data('rle_100_right_ssearch.csv.bz2'),
            '--fasta-out', trimmed,
            '--rle', self.data('rle_100.csv.bz2'),
            '--rle-out', trimmed_rle
        ]

        self.main(args)

        with open(trimmed) as f:
            self.assertEqual(len(list(fastalite(f))), 62)

        with open(trimmed_rle) as f:
            self.assertEqual(len(f.readlines()), 63)
