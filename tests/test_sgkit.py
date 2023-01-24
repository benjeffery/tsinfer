#
# Copyright (C) 2022 University of Oxford
#
# This file is part of tsinfer.
#
# tsinfer is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# tsinfer is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with tsinfer.  If not, see <http://www.gnu.org/licenses/>.
#
"""
Tests for the data files.
"""
import os
import sys
import tempfile

import lmdb
import msprime
import numpy as np
import pytest
import sgkit
import xarray as xr
import zarr

import tsinfer
from tsinfer import exceptions


def open_lmbd_readonly(path):
    # We set the mapsize here because LMBD will map 1TB of virtual memory if
    # we don't, making it hard to figure out how much memory we're actually
    # using.
    map_size = None
    try:
        map_size = os.path.getsize(path)
    except OSError as e:
        raise exceptions.FileFormatError(str(e)) from e
    try:
        store = zarr.LMDBStore(
            path, map_size=map_size, readonly=True, subdir=False, lock=False
        )
    except lmdb.InvalidError as e:
        raise exceptions.FileFormatError(f"Unknown file format:{str(e)}") from e
    except lmdb.Error as e:
        raise exceptions.FileFormatError(str(e)) from e
    return store


def make_ts_and_zarr(path):
    import sgkit.io.vcf

    ts = msprime.sim_ancestry(
        samples=50,
        ploidy=3,
        recombination_rate=0.25,
        sequence_length=50,
        random_seed=42,
    )
    ts = msprime.sim_mutations(
        ts, rate=0.025, model=msprime.BinaryMutationModel(), random_seed=42
    )
    with open(path / "data.vcf", "w") as f:
        ts.write_vcf(f)
    sgkit.io.vcf.vcf_to_zarr(
        # max_alt_alleles=4 tests tsinfer's ability to handle empty string alleles,
        path / "data.vcf",
        path / "data.zarr",
        ploidy=3,
        max_alt_alleles=4,
    )
    return ts, path / "data.zarr"


def add_array_to_dataset(name, array, zarr_path):
    ds = sgkit.load_dataset(zarr_path)
    ds.update({name: xr.DataArray(data=array, dims=["variants"], name=name)})
    sgkit.save_dataset(ds.drop_vars(set(ds.data_vars) - {name}), zarr_path, mode="a")


@pytest.mark.skipif(sys.platform == "win32", reason="No cyvcf2 on windows")
def test_sgkit_dataset(tmp_path):
    ts, zarr_path = make_ts_and_zarr(tmp_path)
    samples = tsinfer.SgkitSampleData(zarr_path)
    inf_ts = tsinfer.infer(samples)
    assert np.array_equal(ts.genotype_matrix(), inf_ts.genotype_matrix())
    # Check that the trees are non-trivial (i.e. the sites have actually been used)
    assert inf_ts.num_trees > 10
    assert inf_ts.num_edges > 200


def test_sgkit_ancestral_allele(tmp_path):
    ts, zarr_path = make_ts_and_zarr(tmp_path)
    ds = sgkit.load_dataset(zarr_path)
    ancestral_allele = ds.variant_allele.values[:, 0]
    ancestral_allele[::2] = ds.variant_allele.values[::2, 1]
    add_array_to_dataset("variant_ancestral_allele", ancestral_allele, zarr_path)
    samples = tsinfer.SgkitSampleData(zarr_path)
    inf_ts = tsinfer.infer(samples)
    for v, inf_v in zip(ts.variants(), inf_ts.variants()):
        assert np.array_equal(
            np.array(v.alleles)[v.genotypes], np.array(inf_v.alleles)[inf_v.genotypes]
        )
    for aa, inf_aa in zip(ancestral_allele, inf_ts.sites()):
        assert aa == inf_aa.ancestral_state
    # TODO test with mask, and with absent alleles etc.


@pytest.mark.skipif(sys.platform == "win32", reason="File permission errors on Windows")
class TestSgkitMask:
    @pytest.mark.parametrize("sites", [[1, 2, 3, 5, 9, 27], [0], []])
    def test_sgkit_variant_mask(self, tmp_path, sites):
        ts, zarr_path = make_ts_and_zarr(tmp_path)
        ds = sgkit.load_dataset(zarr_path)
        sites_mask = np.zeros_like(ds.variant_position, dtype=bool)
        for i in sites:
            sites_mask[i] = True
        add_array_to_dataset("variant_mask", sites_mask, zarr_path)
        samples = tsinfer.SgkitSampleData(zarr_path)
        assert samples.num_sites == len(sites)
        assert np.array_equal(samples.sites_mask, sites_mask)
        assert np.array_equal(
            samples.sites_position, ts.tables.sites.position[sites_mask]
        )
        inf_ts = tsinfer.infer(samples)
        assert np.array_equal(
            ts.genotype_matrix()[sites_mask], inf_ts.genotype_matrix()
        )
        assert np.array_equal(
            ts.tables.sites.position[sites_mask], inf_ts.tables.sites.position
        )
        assert np.array_equal(
            ts.tables.sites.ancestral_state[sites_mask],
            inf_ts.tables.sites.ancestral_state,
        )


@pytest.mark.skipif(sys.platform == "win32", reason="File permission errors on Windows")
def test_sgkit_ancestor(small_sd_fixture, tmp_path):
    with tempfile.TemporaryDirectory(prefix="tsi_eval") as tmpdir:
        f = f"{tmpdir}/test.ancestors"
        tsinfer.generate_ancestors(small_sd_fixture, path=f)
        store = open_lmbd_readonly(f)
        ds = sgkit.load_dataset(store)
        ds = sgkit.variant_stats(ds, merge=True)
        ds = sgkit.sample_stats(ds, merge=True)
        sgkit.display_genotypes(ds)


def test_sgkit_variant_mask(tmp_path):
    ts, zarr_path = make_ts_and_zarr(tmp_path)
    ds = sgkit.load_dataset(zarr_path)
    sites_mask = np.zeros_like(ds["variant_position"], dtype=bool)
    for i in [1, 2, 3, 5, 9, 27]:
        sites_mask[i] = True
    ds.update(
        {
            "variant_mask": xr.DataArray(
                data=sites_mask, dims=["variants"], name="variant_mask"
            )
        }
    )
    sgkit.save_dataset(
        ds.drop_vars(set(ds.data_vars) - {"variant_mask"}), zarr_path, mode="a"
    )
    samples = tsinfer.SgkitSampleData(zarr_path)
    assert samples.num_sites == 6
    assert np.array_equal(samples.sites_mask, sites_mask)
    assert np.array_equal(samples.sites_position, ts.tables.sites.position[sites_mask])
    inf_ts = tsinfer.infer(samples)
    assert np.array_equal(ts.genotype_matrix()[sites_mask], inf_ts.genotype_matrix())
    assert np.array_equal(
        ts.tables.sites.position[sites_mask], inf_ts.tables.sites.position
    )
    assert np.array_equal(
        ts.tables.sites.ancestral_state[sites_mask], inf_ts.tables.sites.ancestral_state
    )
    # TODO - Should test that metadata is correct here


class TestSgkitSampleDataErrors:
    def test_missing_phase(self, tmp_path):
        path = tmp_path / "data.zarr"
        ds = sgkit.simulate_genotype_call_dataset(n_variant=3, n_sample=3)
        sgkit.save_dataset(ds, path)
        with pytest.raises(
            ValueError, match="The call_genotype_phased array is missing"
        ):
            tsinfer.SgkitSampleData(path)

    def test_unphased(self, tmp_path):
        path = tmp_path / "data.zarr"
        ds = sgkit.simulate_genotype_call_dataset(n_variant=3, n_sample=3)
        ds["call_genotype_phased"] = (
            ds["call_genotype"].dims,
            np.zeros(ds["call_genotype"].shape, dtype=bool),
        )
        sgkit.save_dataset(ds, path)
        with pytest.raises(ValueError, match="One or more genotypes are unphased"):
            tsinfer.SgkitSampleData(path)

    def test_phased(self, tmp_path):
        path = tmp_path / "data.zarr"
        ds = sgkit.simulate_genotype_call_dataset(n_variant=3, n_sample=3)
        ds["call_genotype_phased"] = (
            ds["call_genotype"].dims,
            np.ones(ds["call_genotype"].shape, dtype=bool),
        )
        sgkit.save_dataset(ds, path)
        tsinfer.SgkitSampleData(path)

    def test_ploidy1_missing_phase(self, tmp_path):
        path = tmp_path / "data.zarr"
        # Ploidy==1 is always ok
        ds = sgkit.simulate_genotype_call_dataset(n_variant=3, n_sample=3, n_ploidy=1)
        sgkit.save_dataset(ds, path)
        tsinfer.SgkitSampleData(path)

    def test_ploidy1_unphased(self, tmp_path):
        path = tmp_path / "data.zarr"
        ds = sgkit.simulate_genotype_call_dataset(n_variant=3, n_sample=3, n_ploidy=1)
        ds["call_genotype_phased"] = (
            ds["call_genotype"].dims,
            np.zeros(ds["call_genotype"].shape, dtype=bool),
        )
        sgkit.save_dataset(ds, path)
        tsinfer.SgkitSampleData(path)

    def test_empty_alleles_not_at_end(self, tmp_path):
        path = tmp_path / "data.zarr"
        ds = sgkit.simulate_genotype_call_dataset(n_variant=3, n_sample=3, n_ploidy=1)
        ds["variant_allele"] = (
            ds["variant_allele"].dims,
            np.array([["", "C"], ["A", "C"], ["A", "C"]], dtype="S1"),
        )
        sgkit.save_dataset(ds, path)
        with pytest.raises(ValueError, match="Empty alleles must be at the end"):
            samples = tsinfer.SgkitSampleData(path)
            tsinfer.infer(samples)
