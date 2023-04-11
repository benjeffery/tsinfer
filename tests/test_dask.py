import logging
import time

import msprime

import tsinfer
import tsinfer.constants as constants

logger = logging.getLogger(__name__)


def explore():
    from dask.distributed import Client

    # from dask.distributed import LocalCluster

    # cluster = LocalCluster(processes=True, threads_per_worker=1, n_workers=4)
    client = Client("tcp://127.0.0.1:45807")  # noqa

    ts = msprime.sim_ancestry(
        1000, recombination_rate=3e-5, sequence_length=1e7, random_seed=42
    )
    ts = msprime.sim_mutations(ts, rate=3e-5, random_seed=42)
    print(ts)
    sd = tsinfer.SampleData.from_tree_sequence(ts)
    anc = tsinfer.generate_ancestors(sd, num_threads=8)
    print("Ancestors generated")
    anc_ts = tsinfer.match_ancestors(
        sd,
        anc,
        recombination_rate=2e-8,
        precision=13,
        path_compression=True,
        num_threads=8,
    )
    print("Ancestors matched")
    t = time.time()
    inf_ts = tsinfer.match_samples(
        sd,
        anc_ts,
        recombination_rate=2e-8,
        precision=13,
        path_compression=True,
        num_threads=8,
        engine=constants.C_ENGINE,
    )
    print("Samples matched", time.time() - t)
    t = time.time()
    inf_ts_dask = tsinfer.match_samples(
        sd,
        anc_ts,
        recombination_rate=2e-8,
        precision=13,
        path_compression=True,
        num_threads=8,
        engine=constants.C_ENGINE,
        dask=True,
    )
    print("Samples matched dask", time.time() - t)
    inf_ts.tables.assert_equals(inf_ts_dask.tables, ignore_provenance=True)


def test_explore():
    from dask.distributed import Client

    # from dask.distributed import LocalCluster

    # cluster = LocalCluster(processes=True, threads_per_worker=1, n_workers=4)
    client = Client("tcp://127.0.0.1:45807")  # noqa

    ts = msprime.sim_ancestry(
        1000, recombination_rate=3e-5, sequence_length=1e7, random_seed=42
    )
    ts = msprime.sim_mutations(ts, rate=3e-5, random_seed=42)
    print(ts)
    sd = tsinfer.SampleData.from_tree_sequence(ts)
    anc = tsinfer.generate_ancestors(sd, num_threads=8)
    print("Ancestors generated")
    t = time.time()
    anc_ts = tsinfer.match_ancestors(
        sd,
        anc,
        recombination_rate=2e-8,
        precision=13,
        path_compression=True,
        num_threads=8,
    )
    print("Ancestors matched", time.time() - t)
    t = time.time()
    anc_ts_dask = tsinfer.match_ancestors(
        sd,
        anc,
        recombination_rate=2e-8,
        precision=13,
        path_compression=True,
        num_threads=8,
        dask=True,
    )
    print("Ancestors matched dask", time.time() - t)

    anc_ts.tables.assert_equals(anc_ts_dask.tables, ignore_provenance=True)
