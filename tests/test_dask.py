import itertools

import humanize
import msprime
import numpy as np
import tskit
import json
import dask

import tsinfer
import logging
import tsinfer.constants as constants
import tsinfer.algorithm as algorithm

import _tsinfer

logger = logging.getLogger(__name__)


@dask.delayed
def find_path(engine, tree_sequence_builder_wrapper, haplotype, start, end,
              recombination,
              mismatch,
              precision,
              extended_checks,
              ):
    ancestor_matcher_class = _tsinfer.AncestorMatcher if engine == constants.C_ENGINE else algorithm.AncestorMatcher
    matcher = ancestor_matcher_class(
        tree_sequence_builder_wrapper.tsb,
        recombination=recombination,
        mismatch=mismatch,
        precision=precision,
        extended_checks=extended_checks,
    )
    match = np.full(tree_sequence_builder_wrapper.tsb.num_sites, tskit.MISSING_DATA, np.int8)
    missing = haplotype == tskit.MISSING_DATA
    l_r_p = matcher.find_path(haplotype, start, end, match)
    match[missing] = tskit.MISSING_DATA
    diffs = start + np.where(haplotype[start:end] != match[start:end])[0]
    derived_state = haplotype[diffs]
    muts = (diffs.astype(np.int32), derived_state)
    return l_r_p, muts, matcher.mean_traceback_size, matcher.total_memory



sample_data_time_metadata_definition = {
    "description": "Time of an individual from the SampleData file.",
    "type": "number",
    # Defaults aren't currently used, see
    # https://github.com/tskit-dev/tskit/issues/1073
    "default": 0,
}

inference_type_metadata_definition = {
    "description": (
        "The type of inference used at this site. This can be one of the following: "
        f"'{constants.INFERENCE_FULL}' for sites which used the standard tsinfer "
        f"algorithm; '{constants.INFERENCE_NONE}' for sites containing only missing "
        f"data or the ancestral state; '{constants.INFERENCE_PARSIMONY}' for sites "
        "that used a parsimony algorithm to place mutations based on trees inferred "
        "from the remaining data."
    ),
    "type": "string",
    "enum": [
        constants.INFERENCE_NONE,
        constants.INFERENCE_FULL,
        constants.INFERENCE_PARSIMONY,
    ],
}


def _update_site_metadata(current_metadata, inference_type):
    return {"inference_type": inference_type, **current_metadata}


def is_pc_ancestor(flags):
    """
    Returns True if the path compression ancestor flag is set on the specified
    flags value.
    """
    return (flags & constants.NODE_IS_PC_ANCESTOR) != 0


def count_pc_ancestors(flags):
    """
    Returns the number of values in the specified array which have the
    NODE_IS_PC_ANCESTOR set.
    """
    flags = np.array(flags, dtype=np.uint32, copy=False)
    return np.sum(is_pc_ancestor(flags))


def _encode_raw_metadata(obj):
    return json.dumps(obj).encode()


def add_to_schema(schema, name, definition=None, required=False):
    schema = copy.deepcopy(schema)
    if definition is None:
        definition = {}
    try:
        if name in schema["properties"]:
            raise ValueError(f"The metadata {name} is reserved for use by tsinfer")
    except KeyError:
        schema["properties"] = {}
    schema["properties"][name] = definition
    if required:
        if "required" not in schema:
            schema["required"] = []
        schema["required"].append(name)
    return schema


def recombination_rate_to_dist(rho, positions):
    """
    Return the mean number of recombinations between adjacent positions (i.e.
    the genetic distance in Morgans) given either a fixed rate or a RateMap
    """
    try:
        return np.diff(rho.get_cumulative_mass(positions))
    except AttributeError:
        return np.diff(positions) * rho


@staticmethod
def recombination_dist_to_prob(genetic_distances):
    """
    Convert genetic distances (in Morgans) to a probability of recombination,
    (i.e. an odd number of events) assuming a Poisson distribution,
    see Haldane, 1919 J. Genetics 8: 299-309. This maxes out at 0.5 as dist -> inf
    """
    return (1 - np.exp(-genetic_distances * 2)) / 2


@staticmethod
def mismatch_ratio_to_prob(ratio, genetic_distances, num_alleles=2):
    """
    Convert a mismatch ratio, relative to a genetic distance, to a probability
    of mismatch. A mismatch probability of 1 means that the emitted allele has a
    100% probability of being different from the allele implied by the hidden
    state. For all allele types to be emitted with equal probability, regardless
    of the copying haplotype, the mismatch probability should be set to
    1/num_alleles.

    For a small genetic_distance d, setting a ratio of X should give a
    probability of approximately X * r, where r is the recombination probability
    given by recombination_dist_to_prob(d)
    """
    return (1 - np.exp(-genetic_distances * ratio * num_alleles)) / num_alleles


def match_samples(
        sample_data,
        ancestors_ts,
        *,
        recombination_rate=None,
        mismatch_ratio=None,
        path_compression=True,
        indexes=None,
        post_process=None,
        force_sample_times=False,
        num_threads=0,
        # Deliberately undocumented parameters below
        param_recombination=None,  # See :class:`Matcher`
        param_mismatch=None,  # See :class:`Matcher`
        precision=None,
        extended_checks=False,
        engine=constants.C_ENGINE,
        progress_monitor=None,
        simplify=None,  # deprecated
        record_provenance=True,
):
    simplify_only = False  # if true, carry out "old" (deprecated) simplify behaviour
    if simplify is None:
        if post_process is None:
            post_process = True
    else:
        if post_process is not None:
            raise ValueError("Can't specify both `simplify` and `post_process`")
        else:
            if simplify:
                logger.warning(
                    "The `simplify` parameter is deprecated in favour of `post_process`"
                )
                simplify_only = True
                post_process = True
            else:
                post_process = False

    sample_data._check_finalised()
    sample_indexes = tsinfer.check_sample_indexes(sample_data, indexes)
    sample_times = np.zeros(
        len(sample_indexes), dtype=sample_data.individuals_time.dtype
    )
    if force_sample_times:
        individuals = sample_data.samples_individual[:][sample_indexes]
        # By construction all samples in an sd file have an individual: but check anyway
        assert np.all(individuals >= 0)
        sample_times = sample_data.individuals_time[:][individuals]

    # MATCHER INIT
    tables = ancestors_ts.dump_tables()
    inference_site_position = tables.sites.position
    num_sites = len(inference_site_position)
    num_intervals = max(num_sites - 1, 0)
    all_sites = sample_data.sites_position[:]
    index = np.searchsorted(all_sites, inference_site_position)
    num_alleles = sample_data.num_alleles()[index]
    if not np.all(all_sites[index] == inference_site_position):
        raise ValueError(
            "Site positions for inference must be a subset of those in "
            "the sample data file."
        )
    inference_site_id = index

    # Map of site index to tree sequence position. Bracketing
    # values of 0 and L are used for simplicity.
    position_map = np.hstack(
        [inference_site_position, [sample_data.sequence_length]]
    )
    position_map[0] = 0
    recombination = np.zeros(num_sites)  # TODO: reduce len by 1
    mismatch = np.zeros(num_sites)

    if param_recombination is not None or param_mismatch is not None:
        if param_recombination is None or param_mismatch is None:
            raise ValueError(
                "Directly setting probabilities requires specifying "
                "both 'recombination' and 'mismatch'"
            )
        if recombination_rate is not None or mismatch_ratio is not None:
            raise ValueError(
                "Cannot simultaneously specify recombination & recombination_rate, "
                "or mismatch and mismatch_ratio"
            )
        logger.info("Recombination and mismatch probabilities given by user")

    else:
        # Must set recombination and mismatch arrays
        if recombination_rate is None and mismatch_ratio is not None:
            raise ValueError("Cannot use mismatch without setting recombination")
        if (
                recombination_rate is None and mismatch_ratio is None
        ) or num_intervals == 0:
            # Special case: revert to tsinfer 0.1 behaviour with no mismatch allowed
            default_recombination_prob = 1e-2
            default_mismatch_prob = 1e-20  # Substantially < the value above
            param_recombination = np.full(num_intervals, default_recombination_prob)
            param_mismatch = np.full(num_sites, default_mismatch_prob)
            logger.info(
                "Mismatch prevented by setting constant high recombination and "
                + "low mismatch probabilities"
            )
        else:
            genetic_dists = recombination_rate_to_dist(
                recombination_rate, inference_site_position
            )
            param_recombination = recombination_dist_to_prob(genetic_dists)
            if mismatch_ratio is None:
                mismatch_ratio = 1.0
            param_mismatch = np.full(
                num_sites,
                mismatch_ratio_to_prob(
                    mismatch_ratio, np.median(genetic_dists), num_alleles
                ),
            )
            logger.info(
                "Recombination and mismatch probabilities calculated from "
                + f"specified recomb rates with mismatch ratio = {mismatch_ratio}"
            )

    if len(param_recombination) != num_intervals:
        raise ValueError("Bad length for recombination array")
    if len(param_mismatch) != num_sites:
        raise ValueError("Bad length for mismatch array")
    if not (np.all(param_recombination >= 0) and np.all(param_recombination <= 1)):
        raise ValueError("Underlying recombination probabilities not between 0 & 1")
    if not (np.all(param_mismatch >= 0) and np.all(param_mismatch <= 1)):
        raise ValueError("Underlying mismatch probabilities not between 0 & 1")

    if precision is None:
        precision = 13
    recombination[1:] = param_recombination
    mismatch[:] = param_mismatch
    precision = precision

    if len(recombination) == 0:
        logger.info("Fewer than two inference sites: no recombination possible")
    else:
        logger.info(
            "Summary of recombination probabilities between sites: "
            f"min={np.min(recombination):.5g}; "
            f"max={np.max(recombination):.5g}; "
            f"median={np.median(recombination):.5g}; "
            f"mean={np.mean(recombination):.5g}"
        )

    if len(mismatch) == 0:
        logger.info("No inference sites: no mismatch possible")
    else:
        logger.info(
            "Summary of mismatch probabilities over sites: "
            f"min={np.min(mismatch):.5g}; "
            f"max={np.max(mismatch):.5g}; "
            f"median={np.median(mismatch):.5g}; "
            f"mean={np.mean(mismatch):.5g}"
        )
    logger.info(
        f"Matching using {precision} digits of precision in likelihood calcs"
    )

    if engine == constants.C_ENGINE:
        logger.debug("Using C matcher implementation")
        tree_sequence_builder_class = _tsinfer.TreeSequenceBuilder
    elif engine == constants.PY_ENGINE:
        logger.debug("Using Python matcher implementation")
        tree_sequence_builder_class = algorithm.TreeSequenceBuilder
    else:
        raise ValueError(f"Unknown engine:{engine}")

    # Allocate 64K nodes and edges initially. This will double as needed and will
    # quickly be big enough even for very large instances.
    max_edges = 64 * 1024
    max_nodes = 64 * 1024
    allow_multiallele = False
    if np.any(num_alleles > 2) and not allow_multiallele:
        # Currently only used for unsupported extend operation. We can
        # remove in future versions.
        raise ValueError("Cannot currently match with > 2 alleles.")
    tree_sequence_builder = tree_sequence_builder_class(
        num_alleles=num_alleles, max_nodes=max_nodes, max_edges=max_edges
    )
    logger.debug(f"Allocated tree sequence builder with max_nodes={max_nodes}")

    # SAMPLEMATCHER.restore_tree_sequence_builder
    if sample_data.sequence_length != tables.sequence_length:
        raise ValueError(
            "Ancestors tree sequence not compatible: sequence length is different to"
            " sample data file."
        )
    if np.any(tables.nodes.time <= 0):
        raise ValueError("All nodes must have time > 0")

    edges = tables.edges
    # Get the indexes into the position array.
    left = np.searchsorted(position_map, edges.left)
    if np.any(position_map[left] != edges.left):
        raise ValueError("Invalid left coordinates")
    right = np.searchsorted(position_map, edges.right)
    if np.any(position_map[right] != edges.right):
        raise ValueError("Invalid right coordinates")

    # Need to sort by child ID here and left so that we can efficiently
    # insert the child paths.
    index = np.lexsort((left, edges.child))
    nodes = tables.nodes
    tree_sequence_builder.restore_nodes(nodes.time, nodes.flags)
    tree_sequence_builder.restore_edges(
        left[index].astype(np.int32),
        right[index].astype(np.int32),
        edges.parent[index],
        edges.child[index],
    )
    assert tree_sequence_builder.num_match_nodes == 1 + len(
        np.unique(edges.child)
    )

    mutations = tables.mutations
    derived_state = np.zeros(len(mutations), dtype=np.int8)
    mutation_site = mutations.site
    site_id = 0
    mutation_id = 0
    for site in sample_data.sites(inference_site_id):
        while (
                mutation_id < len(mutations) and mutation_site[mutation_id] == site_id
        ):
            allele = mutations[mutation_id].derived_state
            derived_state[mutation_id] = site.reorder_alleles().index(allele)
            mutation_id += 1
        site_id += 1
    tree_sequence_builder.restore_mutations(
        mutation_site, mutations.node, derived_state, mutations.parent
    )
    logger.info(
        "Loaded {} samples {} nodes; {} edges; {} sites; {} mutations".format(
            sample_data.num_samples,
            len(nodes),
            len(edges),
            num_sites,
            len(mutations),
        )
    )

    ########
    sample_id_map = {}

    # SAMPLEMATCHER.match_samples
    for j, t in zip(sample_indexes, sample_times):
        sample_id_map[j] = tree_sequence_builder.add_node(t)

    # SAMPLEMATCHER._match_samples
    if num_sites > 0:
        # SAMPLEMATCHER._match_samples_single_threaded
        sample_haplotypes = sample_data.haplotypes(
            indexes,
            sites=inference_site_id,
            recode_ancestral=True,
        )
        mean_traceback_size = 0
        num_matches = 0
        l_r_p = {}
        muts = {}

        tree_sequence_builder_wrapper = tsinfer.TSBWrapper(engine, tree_sequence_builder, num_alleles, max_nodes, max_edges)
        _, times = tree_sequence_builder.dump_nodes()

        ### SAMPLEMATCHER.__process_sample
        ### MATCHER._find_path
        batch_size = 1000
        start = 0
        end = num_sites

        for j, haplotype in sample_haplotypes:
            l_r_p[j], muts[
                j], matcher_mean_traceback_size, matcher_total_memory = find_path(
                engine, tree_sequence_builder_wrapper,
                haplotype, start, end,
                recombination=recombination,
                mismatch=mismatch,
                precision=precision,
                extended_checks=extended_checks,
                ).compute()

            mean_traceback_size += matcher_mean_traceback_size
            num_matches += 1
            logger.debug(
                "Matched sample {} against {} available;"
                "num_edges={} tb_size={:.2f} match_mem={}".format(
                    j,
                    tree_sequence_builder.num_match_nodes,
                    left.shape[0],
                    matcher_mean_traceback_size,
                    humanize.naturalsize(matcher_total_memory, binary=True),
                )
            )

        for j in sample_indexes:
            node_id = int(sample_id_map[j])
            left, right, parent = l_r_p[j]
            if np.any(times[node_id] > times[parent]):
                p = parent[np.argmin(times[parent])]
                raise ValueError(
                    f"Failed to put sample {j} (node {node_id}) at time "
                    f"{times[node_id]} as it has a younger parent (node {p})."
                )
            tree_sequence_builder.add_path(
                node_id, left, right, parent, compress=path_compression
            )
            diffs, derived_state = muts[j]
            tree_sequence_builder.add_mutations(node_id, diffs, derived_state)

    # SAMPLEMATCHER.get_samples_tree_sequence
    map_additional_sites = True

    schema = sample_data.metadata_schema
    tables.metadata_schema = tskit.MetadataSchema(schema)
    tables.metadata = sample_data.metadata

    schema = sample_data.populations_metadata_schema
    if schema is not None:
        tables.populations.metadata_schema = tskit.MetadataSchema(schema)
    for metadata in sample_data.populations_metadata[:]:
        if schema is None:
            # Use the default json encoding to avoid breaking old code.
            tables.populations.add_row(_encode_raw_metadata(metadata))
        else:
            tables.populations.add_row(metadata)

    schema = sample_data.individuals_metadata_schema
    if schema is not None:
        schema = add_to_schema(
            schema,
            "sample_data_time",
            definition=sample_data_time_metadata_definition,
        )
        tables.individuals.metadata_schema = tskit.MetadataSchema(schema)

    num_ancestral_individuals = len(tables.individuals)
    for ind in sample_data.individuals():
        metadata = ind.metadata
        if ind.time != 0:
            metadata["sample_data_time"] = ind.time
        if schema is None:
            metadata = _encode_raw_metadata(ind.metadata)
        tables.individuals.add_row(
            location=ind.location,
            metadata=metadata,
            flags=ind.flags,
        )

    logger.debug("Adding tree sequence nodes")
    flags, times = tree_sequence_builder.dump_nodes()
    num_pc_ancestors = count_pc_ancestors(flags)

    # All true ancestors are samples in the ancestors tree sequence. We unset
    # the SAMPLE flag but keep other flags intact.
    new_flags = np.bitwise_and(tables.nodes.flags, ~tskit.NODE_IS_SAMPLE)
    tables.nodes.flags = new_flags.astype(np.uint32)
    sample_ids = list(sample_id_map.values())
    assert len(tables.nodes) == sample_ids[0]
    individuals_population = sample_data.individuals_population[:]
    samples_individual = sample_data.samples_individual[:]
    individuals_time = sample_data.individuals_time[:]
    for index, sample_id in sample_id_map.items():
        individual = samples_individual[index]
        if individuals_time[individual] != 0:
            flags[sample_id] = np.bitwise_or(
                flags[sample_id], constants.NODE_IS_HISTORICAL_SAMPLE
            )
        population = individuals_population[individual]
        tables.nodes.add_row(
            flags=flags[sample_id],
            time=times[sample_id],
            population=population,
            individual=num_ancestral_individuals + individual,
        )
    # Add in the remaining non-sample nodes.
    for u in range(sample_ids[-1] + 1, tree_sequence_builder.num_nodes):
        tables.nodes.add_row(flags=flags[u], time=times[u])

    logger.debug("Adding tree sequence edges")
    tables.edges.clear()
    left, right, parent, child = tree_sequence_builder.dump_edges()
    if num_sites == 0:
        # We have no inference sites, so no edges have been estimated. To ensure
        # we have a rooted tree, we add in edges for each sample to an artificial
        # root.
        assert left.shape[0] == 0
        max_node_time = tables.nodes.time.max()
        root = tables.nodes.add_row(flags=0, time=max_node_time + 1)
        ultimate = tables.nodes.add_row(flags=0, time=max_node_time + 2)
        tables.edges.add_row(0, tables.sequence_length, ultimate, root)
        for sample_id in sample_ids:
            tables.edges.add_row(0, tables.sequence_length, root, sample_id)
    else:
        tables.edges.set_columns(
            left=position_map[left],
            right=position_map[right],
            parent=parent,
            child=child,
        )

    logger.debug("Sorting and building intermediate tree sequence.")
    tables.sites.clear()
    tables.mutations.clear()
    tables.sort()

    schema = sample_data.sites_metadata_schema
    if schema is not None:
        schema = add_to_schema(
            schema,
            "inference_type",
            definition=inference_type_metadata_definition,
        )
        tables.sites.metadata_schema = tskit.MetadataSchema(schema)

    # MATCHER.convert_inference_mutations
    mut_site, node, derived_state, _ = tree_sequence_builder.dump_mutations()
    mutation_id = 0
    num_mutations = len(mut_site)
    # progress = self.progress_monitor.get(
    #     "ms_full_mutations", len(self.inference_site_id)
    # )
    schema = tables.sites.metadata_schema.schema
    for site in sample_data.sites(inference_site_id):
        metadata = _update_site_metadata(site.metadata, constants.INFERENCE_FULL)
        if schema is None:
            metadata = _encode_raw_metadata(metadata)
        site_id = tables.sites.add_row(
            site.position,
            ancestral_state=site.ancestral_state,
            metadata=metadata,
        )
        while mutation_id < num_mutations and mut_site[mutation_id] == site_id:
            tables.mutations.add_row(
                site_id,
                node=node[mutation_id],
                derived_state=site.reorder_alleles()[derived_state[mutation_id]],
            )
            mutation_id += 1
    #     progress.update()
    # progress.close()

    ############

    # FIXME this is a shortcut. We should be computing the mutation parent above
    # during insertion (probably)
    tables.build_index()
    tables.compute_mutation_parents()

    logger.info(
        "Built samples tree sequence: {} nodes ({} pc); {} edges; "
        "{} sites; {} mutations".format(
            len(tables.nodes),
            num_pc_ancestors,
            len(tables.edges),
            len(tables.sites),
            len(tables.mutations),
        )
    )

    ts = tables.tree_sequence()
    num_additional_sites = sample_data.num_sites - num_sites
    if map_additional_sites and num_additional_sites > 0:
        logger.info("Mapping additional sites")
        assert np.array_equal(ts.samples(), list(sample_id_map.values()))
        ts = tsinfer.insert_missing_sites(
            sample_data,
            ts,
            sample_id_map=np.array(list(sample_id_map.keys())),
            # progress_monitor=self.progress_monitor,
        )
    else:
        logger.info("Skipping additional site mapping")

    return ts


def test_explore():
    from dask.distributed import Client
    from dask.distributed import LocalCluster
    # cluster = LocalCluster(processes=True, threads_per_worker=1, n_workers=4)
    client = Client('tcp://127.0.0.1:35677')

    ts = msprime.sim_ancestry(1000, recombination_rate=3e-5, sequence_length=1e6,
                              random_seed=42)
    ts = msprime.sim_mutations(ts, rate=3e-5, random_seed=42)
    print(ts)
    sd = tsinfer.SampleData.from_tree_sequence(ts)
    anc = tsinfer.generate_ancestors(sd, num_threads=1)
    anc_ts = tsinfer.match_ancestors(sd, anc, recombination_rate=2e-8, precision=13,
                                     path_compression=True, num_threads=1)
    inf_ts = match_samples(sd, anc_ts, recombination_rate=2e-8, precision=13,
                           path_compression=True, num_threads=1, engine=constants.C_ENGINE)

    # Lets start with match samples - we need to load the ts into the tree sequence builder and then match the samples
    # Lets do that here with simple functions using the algorithm.TreeSequenceBuilder and algorithm.AncestorMatcher

    assert inf_ts.num_sites == sd.num_sites
    assert inf_ts.num_samples == sd.num_samples
    assert np.array_equal(inf_ts.tables.sites.position[:], ts.tables.sites.position[:])
    assert np.array_equal(inf_ts.tables.sites.ancestral_state[:],
                          ts.tables.sites.ancestral_state[:])
    assert np.array_equal(inf_ts.genotype_matrix(), ts.genotype_matrix())
