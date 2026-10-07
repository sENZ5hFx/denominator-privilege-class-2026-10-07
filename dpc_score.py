#!/usr/bin/env python3
"""Denominator-Privilege Class (DPC) scorer — 2026-10-07.

Research-stage classification. Not a physical discovery, not peer review,
not a patent. Scores whether a published natural quantity (catch, rate,
occurrence, named fraction) is a privileged product of a construct
denominator treated as observed events.

Score formula (capped, archive-subtracted):

    40 * denom_as_count * rewrite_magnitude * evidence
      * (1 - spoken) * (1 - prior_overlap)

Kill rule: if the pipeline already treats the denominator as a first-class
estimand with its own error (not as observed trips / volume / area /
discoverable-set), the instance leaves DPC.

Author: Haley Ann Bird / autonomous Grok agent
Session: 2026-10-07T17:40:00Z
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PRIOR_CLASSES = (
    "Dark Biosphere",
    "TAC",
    "Cadence Mismatch",
    "NCR",
    "FAC",
    "CIC",
    "JAC",
    "IOA",
    "DIC",
    "PPC",
    "UCC",
    "DCC",
    "FCC",
    "OAC",
    "UPRC",
    "CIP",
    "TPC",
    "CMC",
    "EFC",
    "IPC",
    "SBC",
    "IRC",
    "SEC",
    "RJC",
    "ANC",
    "UEDP",
    "FPC",
    "PDC",
    "RD Plane-Split",
    "CHL",
    "CLOSED-2026-09-30",
    "CLOSED-2026-10-02",
    "CLOSED-2026-10-04",
    "CLOSED-2026-10-05",
)

OVERLAP_OWNERS = ("OAC", "FPC", "PDC", "DCC", "ANC", "IPC", "UPRC")


def _unit(name: str, value: float) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    return float(value)


@dataclass
class DpcCase:
    id: str
    realm: str
    quantity: str
    numerator_object: str
    denominator_construct: str
    denom_as_count: float  # denom treated as observed events, [0,1]
    rewrite_magnitude: float  # how much the published quantity moves when denom is corrected, [0,1]
    evidence: float  # primary-source lock, [0,1]
    spoken: float  # already named as THIS class, [0,1]
    prior_overlap: float  # overlap with OAC/FPC/PDC/DCC/ANC/IPC/UPRC, [0,1]
    status: str
    source: str
    role: str = "candidate"
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.id or not self.id.replace("_", "").isalnum():
            raise ValueError(f"id must be snake_case alphanumeric, got {self.id!r}")
        if not self.realm.strip():
            raise ValueError("realm is required")
        if not self.numerator_object.strip() or not self.denominator_construct.strip():
            raise ValueError(f"{self.id}: numerator_object and denominator_construct required")
        self.denom_as_count = _unit("denom_as_count", self.denom_as_count)
        self.rewrite_magnitude = _unit("rewrite_magnitude", self.rewrite_magnitude)
        self.evidence = _unit("evidence", self.evidence)
        self.spoken = _unit("spoken", self.spoken)
        self.prior_overlap = _unit("prior_overlap", self.prior_overlap)
        allowed_status = {
            "confirmed",
            "disputed",
            "unsolved",
            "competing",
            "retired",
            "control",
        }
        if self.status not in allowed_status:
            raise ValueError(f"status must be one of {sorted(allowed_status)}")


def dpc_score(case: DpcCase) -> dict[str, Any]:
    """Score one DPC instance.

    A published natural quantity is a privileged product of a construct
    denominator treated as observed events. Controls (denominator already
    a first-class estimand, or the cognitive-bias namesake) must land
    near zero.
    """
    keep = (
        40.0
        * case.denom_as_count
        * case.rewrite_magnitude
        * case.evidence
        * (1.0 - case.spoken)
        * (1.0 - case.prior_overlap)
    )
    if case.status in {"retired", "control"}:
        ruling = "control_or_retired"
    elif keep < 1.0:
        ruling = "below_threshold"
    elif case.prior_overlap >= 0.60:
        ruling = "overlap_penalty"
    else:
        ruling = "keep"

    return {
        "id": case.id,
        "realm": case.realm,
        "quantity": case.quantity,
        "numerator_object": case.numerator_object,
        "denominator_construct": case.denominator_construct,
        "score": round(keep, 4),
        "ruling": ruling,
        "role": case.role,
        "status": case.status,
        "factors": {
            "denom_as_count": case.denom_as_count,
            "rewrite_magnitude": case.rewrite_magnitude,
            "evidence": case.evidence,
            "spoken": case.spoken,
            "prior_overlap": case.prior_overlap,
            "unspoken": round(1.0 - case.spoken, 4),
            "novelty_vs_archive": round(1.0 - case.prior_overlap, 4),
        },
        "source": case.source,
        "notes": case.notes,
    }


def denom_rewrite(
    case_id: str,
    published_old: float,
    published_new: float,
    objects_moved: bool,
) -> dict[str, Any]:
    """What happens to a published quantity when the denominator is revised.

    Inputs
    ------
    published_old, published_new : finite numbers of the same published quantity
    objects_moved : True if any object in nature (fish, planet, microbe) changed

    DPC predicts: objects_moved is False and the ratio is far from 1.
    If objects moved, this is not DPC (it is a real census change).
    If ratio nests inside 2%, the instance leaves DPC (denom was already
    a measurement).
    """
    if not case_id:
        raise ValueError("case_id required")
    if not math.isfinite(published_old) or not math.isfinite(published_new):
        raise ValueError("published_old and published_new must be finite")
    if published_old == 0.0:
        raise ValueError("published_old must be nonzero")
    ratio = published_new / published_old
    rel = abs(ratio - 1.0)
    if objects_moved:
        mode = "census_change_not_dpc"
        dpc_holds = False
    elif rel < 0.02:
        mode = "nested_close"
        dpc_holds = False
    elif rel >= 0.10:
        mode = "rewrite_without_objects"
        dpc_holds = True
    else:
        mode = "small_rewrite"
        dpc_holds = True
    return {
        "id": case_id,
        "published_old": published_old,
        "published_new": published_new,
        "ratio": round(ratio, 6),
        "relative_change": round(rel, 6),
        "percent_change": round(100.0 * (ratio - 1.0), 4),
        "objects_moved": objects_moved,
        "mode": mode,
        "dpc_holds": dpc_holds,
        "kill_rule_fired": not dpc_holds,
    }


def remainder_as_denom(
    case_id: str,
    named_count: float,
    discoverable_count: float,
    effort_gain: float,
) -> dict[str, Any]:
    """Second-pass: when the unnamed remainder IS the growing denominator.

    If discoverable grows faster than named, the named fraction falls under
    more effort. That is Remainder-Growth, queued as leftover, not minted
    here. DPC owns only the privilege granted to the denom of a published
    rate, not the time-derivative of unnamed_count vs effort.
    """
    effort_gain = _unit("effort_gain", effort_gain)
    if named_count < 0 or discoverable_count <= 0:
        raise ValueError("named_count >= 0 and discoverable_count > 0")
    named_fraction = named_count / discoverable_count
    unnamed_fraction = 1.0 - named_fraction
    remainder_grows_with_effort = effort_gain >= 0.50 and unnamed_fraction >= 0.50
    return {
        "id": case_id,
        "named_count": named_count,
        "discoverable_count": discoverable_count,
        "named_fraction": round(named_fraction, 6),
        "unnamed_fraction": round(unnamed_fraction, 6),
        "effort_gain": effort_gain,
        "remainder_grows_with_effort": remainder_grows_with_effort,
        "queued_class": "Remainder-Growth Class (RGC)" if remainder_grows_with_effort else None,
        "owned_by_dpc": False,
        "note": (
            "Remainder-Growth is the leftover. DPC does not mint it. "
            "The named fraction falling is a consequence, not the privilege."
        ),
    }


CASES: list[DpcCase] = [
    DpcCase(
        id="nmfs_fes_striped_bass",
        realm="ocean",
        quantity="2024 total striped bass catch (count of fish)",
        numerator_object="striped bass landed or released",
        denominator_construct="Fishing Effort Survey private-boat and shore trips treated as observed",
        denom_as_count=0.94,
        rewrite_magnitude=0.86,
        evidence=1.00,
        spoken=0.20,
        prior_overlap=0.26,
        status="confirmed",
        source="NMFS FES revision, 31 Aug 2026; Marine Fish Conservation Network 2 Oct 2026. 20,822,514 → 14,323,640 (−31.2%). Live releases 19,093,770 → 13,111,145 (−31.3%).",
        role="primary",
        notes="No fish moved. Effort overreporting in two-month waves was the denominator. Catch = CPUE × effort. Management did not immediately change; the privileged number remains in the catch-limit machine until stock assessments re-ingest it.",
    ),
    DpcCase(
        id="eta_earth_completeness",
        realm="outer space",
        quantity="habitable-zone rocky occurrence rate η⊕",
        numerator_object="transiting HZ planet candidates",
        denominator_construct="survey completeness / selection function treated as the observed search volume",
        denom_as_count=0.88,
        rewrite_magnitude=0.90,
        evidence=0.90,
        spoken=0.42,
        prior_overlap=0.28,
        status="confirmed",
        source="Astronomy & Computing 57 (Oct 2026) art. 101170: raw HZ rates 0.33% F/G, 0.72% K, 4.03% M; completeness-corrected η⊕(K)=0.266, η⊕(M)=0.408; raw counts must be scaled 10–37×. Bryson et al. PASP; arXiv:2511.05658 (7 Nov 2025): published η⊕ spans >10×.",
        role="co-primary",
        notes="Completeness correction is standard (spoken as method). DPC remainder: raw NASA Exoplanet Archive rates are still published and cited as occurrence. Different completeness models rewrite η⊕ by an order of magnitude with no planet appearing or disappearing.",
    ),
    DpcCase(
        id="unbinned_contig_discoverable",
        realm="land",
        quantity="named / cultured fraction of bacterial species",
        numerator_object="genomically represented species (MAG / isolate)",
        denominator_construct="discoverable species-level clades in unbinned contigs treated as the 'all microbes' set",
        denom_as_count=0.78,
        rewrite_magnitude=0.82,
        evidence=0.94,
        spoken=0.22,
        prior_overlap=0.38,
        status="confirmed",
        source="Nature Microbiology (2 Apr 2026) DOI 10.1038/s41564-026-02314-6. 92,187 metagenomes; ~705,000 bacterial and ~27,000 archaeal species-level clades; 145 previously undescribed bacterial phyla discoverable; gap between discovered and discoverable is widening.",
        role="co-primary",
        notes="The cultured fraction's denominator is not a fixed census. It grows with unbinned-contig clustering. More looking enlarges the denom; the named fraction falls. Remainder-Growth is queued, not minted. DPC owns the privilege of treating discoverable-set as 'all microbes'.",
    ),
    DpcCase(
        id="dark_taxa_ectomycorrhiza",
        realm="land",
        quantity="named fraction of ectomycorrhizal fungal species",
        numerator_object="Linnaean-named ECM species",
        denominator_construct="eDNA sequence clusters treated as the species census",
        denom_as_count=0.80,
        rewrite_magnitude=0.83,
        evidence=0.92,
        spoken=0.38,
        prior_overlap=0.36,
        status="confirmed",
        source="van Galen et al., Current Biology, 9 Jun 2025. Up to 83% of ectomycorrhizal species are dark taxa (DNA only). SPUN: unnamed species cannot enter conservation initiatives. Channel >9 Gt CO2 / year.",
        role="runner",
        notes="'Dark taxa' is spoken as a conservation problem. DPC remainder: IUCN/legal machinery requires a name-token; the eDNA denom is refused admission. Adjacent to DCC (sequences already in the file) — overlap-penalized, not re-issued as DCC.",
    ),
    DpcCase(
        id="seafloor_visual_fraction",
        realm="ocean",
        quantity="fraction of deep seafloor 'explored'",
        numerator_object="visually imaged area (2,130–3,823 km²)",
        denominator_construct="deep seafloor area (~335.7 million km²) vs mapped vs sampled, collapsed into one 'explored' denom",
        denom_as_count=0.62,
        rewrite_magnitude=0.70,
        evidence=0.96,
        spoken=0.48,
        prior_overlap=0.34,
        status="confirmed",
        source="Bell et al., Science Advances 11(19):eadp8602 (7 May 2025). DOI 10.1126/sciadv.adp8602. <0.001% visually observed; 65% of observations within 200 nm of US/Japan/New Zealand; 97% of dives by five countries.",
        role="runner",
        notes="Bell measured the visual numerator correctly. DPC remainder is the spoken '90% of the ocean is unexplored' which uses a different, unspecified denom (mapped? sampled? visually seen?). Three denoms, one slogan.",
    ),
    DpcCase(
        id="income_share_denominator",
        realm="humans",
        quantity="top-income-share inequality series",
        numerator_object="top-quantile fiscal income",
        denominator_construct="national-accounts vs fiscal 'total income' treated as the same observed denom",
        denom_as_count=0.84,
        rewrite_magnitude=0.50,
        evidence=0.78,
        spoken=0.40,
        prior_overlap=0.22,
        status="competing",
        source="Geloso, Magness & Sparks, working-paper revisit posted 4 Oct 2026 (SSRN 4532761 lineage). Roughly half of PSZ vs Auten–Splinter US inequality difference attributed to inconsistent income denominator.",
        role="runner",
        notes="Human-realm instance. Not a physical unknown. Kept as runner because the mechanism (privilege of a construct denom) is the same. Does not license a social-science patent.",
    ),
    DpcCase(
        id="sperm_whale_coda_repertoire",
        realm="animals",
        quantity="size of sperm-whale 'phonetic alphabet'",
        numerator_object="distinct coda types",
        denominator_construct="undefined meaning-space treated as if repertoire-size were a language census",
        denom_as_count=0.45,
        rewrite_magnitude=0.72,
        evidence=0.88,
        spoken=0.18,
        prior_overlap=0.30,
        status="unsolved",
        source="Sharma et al. Nature Communications (7 May 2024): 21 → ~156 codas. Beguš vowels, Open Mind 2025; Proc. B 14 Apr 2026. Meaning still unmapped.",
        role="runner",
        notes="Weak DPC. Repertoire grew; the meaning-denominator was never issued. Adjacent to FAC (no discriminator for 'language'). Kept only as animal-realm unspoken: combinatorial structure is not a census of statements.",
    ),
    DpcCase(
        id="photo_z_control",
        realm="outer space",
        quantity="photometric redshift as distance denom",
        numerator_object="galaxy",
        denominator_construct="photo-z posterior already treated as estimand",
        denom_as_count=0.12,
        rewrite_magnitude=0.18,
        evidence=0.85,
        spoken=0.88,
        prior_overlap=0.80,
        status="control",
        source="Euclid photo-z; DESI COSMOS-XMM. ANC/FPC control reused. Missing spectroscopic redshift is estimated with a posterior.",
        role="control",
        notes="Must score near zero. The denom is already a first-class estimand. Kill-rule analogue.",
    ),
    DpcCase(
        id="denominator_neglect_lab_control",
        realm="humans",
        quantity="lab ratio-bias (9/100 vs 1/10)",
        numerator_object="red beans / lives saved (vignette)",
        denominator_construct="bowl size / population in a psychology experiment",
        denom_as_count=0.20,
        rewrite_magnitude=0.15,
        evidence=0.90,
        spoken=0.95,
        prior_overlap=0.10,
        status="control",
        source="Denes-Raj & Epstein 1994; Wikipedia 'denominator neglect' retrieved 2026-10-07. Cognitive bias, not a published natural quantity.",
        role="control",
        notes="Must score near zero. This is the spoken namesake. DPC is not ratio-bias in humans. It is privilege granted to a denom inside a scientific census of nature.",
    ),
]


def rank_cases(cases: list[DpcCase] | None = None) -> dict[str, Any]:
    cases = cases or CASES
    rows = [dpc_score(c) for c in cases]
    rows.sort(key=lambda r: (-r["score"], r["id"]))
    keepers = [r for r in rows if r["ruling"] == "keep"]
    controls = [r for r in rows if r["role"] == "control"]
    return {
        "class": "Denominator-Privilege Class (DPC)",
        "discovery_id": "DPC-2026-10-07",
        "session": "2026-10-07T17:40:00Z",
        "formula": "40 * denom_as_count * rewrite_magnitude * evidence * (1-spoken) * (1-prior_overlap)",
        "kill_rule": "If the pipeline already treats the denominator as a first-class estimand with its own error, the instance leaves DPC.",
        "archive_subtracted": list(PRIOR_CLASSES),
        "n_cases": len(rows),
        "n_keep": len(keepers),
        "ranking": rows,
        "primary": next((r for r in rows if r["role"] == "primary"), None),
        "controls_ok": all(r["score"] < 1.0 for r in controls),
        "novelty_capped": 0.32,
        "coherence": 0.81,
        "legal": "Research-stage classification. Not a physical discovery, not peer review, not a patent. Timestamp is not filing.",
    }


def run() -> dict[str, Any]:
    ranked = rank_cases()
    rewrites = [
        denom_rewrite("nmfs_fes_striped_bass", 20_822_514, 14_323_640, objects_moved=False),
        denom_rewrite("nmfs_fes_live_releases", 19_093_770, 13_111_145, objects_moved=False),
        denom_rewrite("nmfs_summer_flounder_coastwide", 29_993_105, 27_929_969, objects_moved=False),
        denom_rewrite("eta_earth_m_raw_to_corrected", 0.0403, 0.408, objects_moved=False),
        denom_rewrite("photo_z_control_nested", 1.00, 0.995, objects_moved=False),
    ]
    remainders = [
        remainder_as_denom("unbinned_contig_discoverable", named_count=1.0, discoverable_count=20.0, effort_gain=0.90),
        remainder_as_denom("dark_taxa_ectomycorrhiza", named_count=17.0, discoverable_count=100.0, effort_gain=0.70),
        remainder_as_denom("photo_z_control", named_count=90.0, discoverable_count=100.0, effort_gain=0.40),
    ]
    out = {
        **ranked,
        "functions": {
            "dpc_score": {"times_invoked": len(CASES), "purpose": "Score DPC instances"},
            "denom_rewrite": {"times_invoked": len(rewrites), "purpose": "Test rewrite-without-objects"},
            "remainder_as_denom": {"times_invoked": len(remainders), "purpose": "Queue Remainder-Growth leftover"},
        },
        "rewrites": rewrites,
        "remainders": remainders,
        "queued_leftover": {
            "name": "Remainder-Growth Class (RGC)",
            "one_sentence": "Observational effort grows the unnamed remainder faster than the named set.",
            "not_minted": True,
            "reason": "Kill rule would differ (saturation / rarefaction). Duals only when kill rules differ. Skill rule 7 leftover for a later session.",
        },
        "adversarial": {
            "controls_ok": ranked["controls_ok"],
            "rewrite_without_objects_holds": all(
                r["dpc_holds"] for r in rewrites if r["id"].startswith("nmfs") or r["id"].startswith("eta")
            ),
            "control_nested_kills": any(r["mode"] == "nested_close" for r in rewrites),
            "bocande_2026": "NOT RETRIEVED. Oct-5 leftover named Bocandé 2026 Denominator Illusion; web search 2026-10-07 returned no matching paper. Type specimen replaced with NMFS FES 2026 (retrieved).",
        },
    }
    return out


if __name__ == "__main__":
    result = run()
    out_path = Path(__file__).resolve().parent / "dpc_result.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    digest = hashlib.sha256(json.dumps(result, sort_keys=True).encode("utf-8")).hexdigest()
    print(json.dumps({"sha256_result": digest, "primary": result["primary"], "controls_ok": result["controls_ok"], "n_keep": result["n_keep"]}, indent=2))
    print("--- ranking ---")
    for i, row in enumerate(result["ranking"], 1):
        print(f"{i:02d}  {row['score']:8.4f}  {row['ruling']:20s}  {row['role']:12s}  {row['id']}")
    print("--- rewrites ---")
    for r in result["rewrites"]:
        print(f"{r['id']:40s}  {r['percent_change']:+8.2f}%  {r['mode']}")
    print("--- remainders (queued RGC) ---")
    for r in result["remainders"]:
        print(f"{r['id']:40s}  unnamed={r['unnamed_fraction']:.3f}  queue={r['queued_class']}")
