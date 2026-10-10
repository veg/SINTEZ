"""Tests for SINTEZ CLI and GFF3 output formatting."""

import tempfile
from pathlib import Path
import pytest
from sintez.cli import write_gff3


def test_write_gff3():
    preds = [
        {
            "spans": [(100, 300), (400, 600)],
            "strand": "+",
            "score": 42.5,
        },
        {
            "spans": [(800, 1200)],
            "strand": "-",
            "score": 38.0,
        },
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        out_gff = Path(tmpdir) / "test_out.gff3"
        write_gff3(preds, chrom_id="chrTest", out_path=out_gff)

        assert out_gff.exists()
        content = out_gff.read_text()

        # Check headers
        assert "##gff-version 3" in content
        assert "##sequence-region chrTest 1 100000000" in content

        # Check features
        assert "SINTEZ_00001" in content
        assert "SINTEZ_00002" in content
        assert "chrTest\tSINTEZ\tgene\t101\t600\t42.50\t+\t.\tID=SINTEZ_00001" in content
        assert "chrTest\tSINTEZ\tCDS\t101\t300\t.\t+\t0\tID=SINTEZ_00001.1.cds1" in content
        assert "chrTest\tSINTEZ\tCDS\t401\t600\t.\t+\t0\tID=SINTEZ_00001.1.cds2" in content
        assert "chrTest\tSINTEZ\tgene\t801\t1200\t38.00\t-\t.\tID=SINTEZ_00002" in content
        assert "biotype=protein_coding" in content


def test_write_gff3_stratified():
    preds = [
        {
            "spans": [(100, 300)],
            "strand": "+",
            "score": 30.0,
            "biotype": "transposon_derived",
            "gene_type": "mobile_element",
            "te_family": "LTR_Retrotransposon/Gypsy",
        }
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        out_gff = Path(tmpdir) / "test_te.gff3"
        write_gff3(preds, chrom_id="chrTest", out_path=out_gff)
        content = out_gff.read_text()
        assert "biotype=transposon_derived" in content
        assert "gene_type=mobile_element" in content
        assert "te_family=LTR_Retrotransposon/Gypsy" in content

