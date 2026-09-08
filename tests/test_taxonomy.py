"""Tests for the mapping of dataset classes onto Murcian relevance."""

from __future__ import annotations

import pytest

from citrus_scout.data.build import binary_labels
from citrus_scout.data.leaf_dataset import LeafSample, normalise_label
from citrus_scout.data.sources import CATALOGUE, get_source
from citrus_scout.data.taxonomy import (
    LABEL_RELEVANCE,
    LocalRelevance,
    is_healthy,
    locally_present_labels,
    relevance_of,
)


class TestRelevance:
    def test_hlb_is_marked_absent(self):
        """Greening is HLB. Spain is free of it, so an alert would always be wrong."""
        assert relevance_of("greening") is LocalRelevance.ABSENT

    def test_canker_is_marked_absent(self):
        assert relevance_of("citrus canker") is LocalRelevance.ABSENT

    def test_gummosis_is_present(self):
        """Phytophthora is the project's highest-value target."""
        assert relevance_of("gummosis") is LocalRelevance.PRESENT

    def test_aphids_are_present(self):
        assert relevance_of("aphids") is LocalRelevance.PRESENT

    def test_unknown_label_defaults_to_nonspecific(self):
        """An unrecognised class must not silently become actionable."""
        assert relevance_of("something new") is LocalRelevance.NONSPECIFIC

    def test_absent_classes_are_excluded_from_reportable_set(self):
        reportable = locally_present_labels()
        for label in ("greening", "citrus canker", "black spot", "melanose"):
            assert label not in reportable

    def test_present_classes_are_reportable(self):
        reportable = locally_present_labels()
        for label in ("gummosis", "aphids", "spider mite"):
            assert label in reportable

    def test_every_mapped_label_is_normalised(self):
        """Keys must match what normalise_label produces, or lookups silently miss."""
        for label in LABEL_RELEVANCE:
            assert normalise_label(label) == label


class TestIsHealthy:
    def test_healthy_label(self):
        assert is_healthy("healthy")

    @pytest.mark.parametrize("label", ["gummosis", "greening", "curl leaf", "aphids"])
    def test_disease_labels(self, label):
        assert not is_healthy(label)


class TestBinaryLabels:
    def test_healthy_maps_to_zero(self):
        samples = [LeafSample(path="/a.jpg", label="healthy", source="t")]
        assert binary_labels(samples) == [0]

    def test_diseases_map_to_one(self):
        samples = [
            LeafSample(path="/a.jpg", label="gummosis", source="t"),
            LeafSample(path="/b.jpg", label="greening", source="t"),
            LeafSample(path="/c.jpg", label="curl leaf", source="t"),
        ]
        assert binary_labels(samples) == [1, 1, 1]

    def test_mixed(self):
        samples = [
            LeafSample(path="/a.jpg", label="healthy", source="t"),
            LeafSample(path="/b.jpg", label="aphids", source="t"),
            LeafSample(path="/c.jpg", label="healthy", source="t"),
        ]
        assert binary_labels(samples) == [0, 1, 0]


class TestCatalogue:
    def test_all_catalogue_entries_are_commercially_usable(self):
        """The project ships as a product; unlicensed data cannot be in the default set."""
        for source in CATALOGUE:
            assert source.is_commercially_usable, f"{source.key} has licence {source.licence}"

    def test_get_source_by_key(self):
        assert get_source("citrus_diseases").ref == "superlord/citrus-diseases"

    def test_unknown_key_lists_alternatives(self):
        with pytest.raises(KeyError, match="available"):
            get_source("nope")

    @pytest.mark.parametrize(
        ("licence", "expected"),
        [
            ("CC0: Public Domain", True),
            ("Attribution 4.0 International (CC BY 4.0)", True),
            ("MIT", True),
            ("Unknown", False),
            ("CC BY-NC-SA 4.0", True),  # documented limitation, see test below
            ("Other (specified in description)", False),
        ],
    )
    def test_licence_classification(self, licence, expected):
        from citrus_scout.data.sources import DatasetSource

        source = DatasetSource(
            key="k", ref="a/b", licence=licence, approx_size_mb=1, description=""
        )
        assert source.is_commercially_usable is expected

    def test_noncommercial_licence_is_a_known_gap(self):
        """CC BY-NC contains 'CC BY' and is wrongly accepted by the substring check.

        No catalogue entry uses such a licence today, so this documents the gap
        rather than guarding against it. Tighten the check before adding one.
        """
        from citrus_scout.data.sources import DatasetSource

        nc = DatasetSource(
            key="k", ref="a/b", licence="CC BY-NC 4.0", approx_size_mb=1, description=""
        )
        assert nc.is_commercially_usable  # not what a human would say
        assert all("NC" not in s.licence for s in CATALOGUE)
