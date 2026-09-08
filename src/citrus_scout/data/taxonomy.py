"""Mapping from public dataset labels onto targets that matter in Murcia.

The public datasets were assembled elsewhere, mostly in regions where citrus faces
different pressures than the Spanish Levante. Their class lists reflect that: canker,
greening (HLB), black spot and melanose dominate, and none of those occur in Spain.

Training on all of them anyway is still the right call at this stage, because the
backbone learns citrus leaf texture and lesion morphology in general. What must not
happen is shipping a product that reports "greening detected" on a Murcian grove,
where the bacterium is absent and the alert is guaranteed wrong.

So labels are tagged by local relevance, and that tag drives what the deployed
system is allowed to report.
"""

from __future__ import annotations

from enum import StrEnum


class LocalRelevance(StrEnum):
    """Whether a class corresponds to something that occurs in Murcia."""

    PRESENT = "present"
    """Occurs in Murcia. Detections are actionable."""

    ABSENT = "absent"
    """Does not occur in Spain. Useful for pre-training, never for alerting."""

    NONSPECIFIC = "nonspecific"
    """Real but with many possible causes; a symptom rather than a diagnosis."""

    HEALTHY = "healthy"


# Label to relevance. Keys are normalised labels as produced by
# citrus_scout.data.leaf_dataset.normalise_label.
LABEL_RELEVANCE: dict[str, LocalRelevance] = {
    # Present in Murcia and directly actionable.
    "gummosis": LocalRelevance.PRESENT,
    "aphids": LocalRelevance.PRESENT,
    "leaf minnor": LocalRelevance.PRESENT,  # dataset's spelling of leaf miner
    "spider mite": LocalRelevance.PRESENT,
    "citrus mite": LocalRelevance.PRESENT,
    "sooty mould": LocalRelevance.PRESENT,
    # Absent from Spain. Kept for pre-training only.
    "citrus canker": LocalRelevance.ABSENT,
    "greening": LocalRelevance.ABSENT,  # HLB
    "black spot": LocalRelevance.ABSENT,
    "melanose": LocalRelevance.ABSENT,
    "anthracnose": LocalRelevance.ABSENT,
    "bacterial blight": LocalRelevance.ABSENT,
    "curl virus": LocalRelevance.ABSENT,
    # Real symptoms, many possible causes.
    "curl leaf": LocalRelevance.NONSPECIFIC,
    "dry leaf": LocalRelevance.NONSPECIFIC,
    "deficiency leaf": LocalRelevance.NONSPECIFIC,
    "healthy": LocalRelevance.HEALTHY,
}

# Notes explaining why a class carries its tag, surfaced in reports and docs.
RELEVANCE_NOTES: dict[str, str] = {
    "gummosis": (
        "Phytophthora. The highest-value target for this project: it is present in "
        "Murcia and its canopy decline is visible from the screening pass."
    ),
    "greening": (
        "HLB. Spain is free of Candidatus Liberibacter and of Diaphorina citri, and "
        "Trioza erytreae has not reached the Mediterranean Levante. No local ground "
        "truth is obtainable, and any alert in Murcia is a false positive."
    ),
    "citrus canker": "Quarantine pathogen, absent from Spain.",
    "black spot": "Absent from Spanish groves.",
    "aphids": "Present in spring flush. Symptom is on young shoots, so close-range only.",
    "spider mite": "Tetranychus urticae, present and damaging in late summer.",
    "citrus mite": "Present. Individual mites are sub-millimetre; damage is what shows.",
    "leaf minnor": "Citrus leafminer. Galleries are under 1 mm wide, so close-range only.",
    "sooty mould": "Follows honeydew from whitefly or scale. Visible as canopy darkening.",
}


def relevance_of(label: str) -> LocalRelevance:
    """Relevance tag for a normalised label, defaulting to nonspecific."""
    return LABEL_RELEVANCE.get(label, LocalRelevance.NONSPECIFIC)


def locally_present_labels() -> frozenset[str]:
    """Labels the deployed system may legitimately report in Murcia."""
    return frozenset(
        label
        for label, relevance in LABEL_RELEVANCE.items()
        if relevance in (LocalRelevance.PRESENT, LocalRelevance.HEALTHY)
    )


def is_healthy(label: str) -> bool:
    """Whether a label denotes a healthy leaf.

    Used to collapse the multi-class problem into the binary healthy/affected task
    the product actually needs.
    """
    return relevance_of(label) is LocalRelevance.HEALTHY
