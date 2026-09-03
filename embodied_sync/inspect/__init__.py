"""Human-verifiable evidence for one event-train alignment.

The rest of the library measures alignments; this subpackage is where a
*person* checks one. It exists because a wrong event pairing can be
internally self-consistent: when the matcher pairs event *i* with event
*i±1*, downstream fit statistics can agree with the wrong answer.

So the page this subpackage renders does not report a better statistic.
It renders the chosen pairing beside the two the matcher rejected, at
the instants those hypotheses predict, and asks a person whether the
chosen column depicts the same moment as the reference and the
neighbours do not. That comparison is the product; everything else on
the page is context for it.

Where the dataset goes
----------------------
Behind :class:`~embodied_sync.inspect.provider.EvidenceProvider`. Media
decoding — ffmpeg, HDF5, whatever a rig needs — lives in the caller's
provider, never here, so importing this subpackage costs a base install
(numpy) and nothing else. Other datasets implement the same small provider
contract without changing the inspector.

:mod:`~embodied_sync.inspect.lerobot_pts` is the same idea aimed at a
dataset rather than an alignment: it reproduces LeRobot's video timestamp
check with full precision and reports which of several failure modes with
one shared error message a local v3 dataset actually has.

Typical use::

    inspection = inspection_from_alignment(alignment, events_a, events_b)
    evidence = collect_evidence(inspection, reference_provider, candidate_provider)
    page = build_page(inspection, evidence, PageContext(...))
"""

from embodied_sync.inspect.evidence import (
    AMBIGUOUS_MARGIN,
    DEFAULT_MAX_EVENTS,
    INDISTINCT_GRAY_LEVELS,
    Candidate,
    EventEvidence,
    Inspection,
    collect_evidence,
    inspection_from_alignment,
    perturb,
    residuals_ns,
    restrict_to_overlap,
)
from embodied_sync.inspect.lerobot_pts import (
    DEFAULT_TOLERANCE_S,
    BoundaryCheck,
    DatasetAudit,
    EpisodeTrend,
    LeRobotPTSAuditError,
    PTSQuery,
    VideoAudit,
    VideoEvidence,
    audit_lerobot_dataset,
    classify_video,
    float32_ulp_ns,
    legacy_float32_residual_ns,
    precision_band_ns,
    render_audit_text,
)
from embodied_sync.inspect.provider import (
    AudioClip,
    BaseProvider,
    ChannelReading,
    EvidenceProvider,
    FrameImage,
    NullProvider,
    StreamInfo,
)
from embodied_sync.inspect.render import PageContext, build_page
from embodied_sync.inspect.verification import (
    AlignmentVerifier,
    HTTPAlignmentVerifier,
    VerificationRequest,
    VerificationResult,
    VerificationReview,
    VerificationServiceError,
    apply_verification_review,
    review_verification,
    verification_document,
    verify_alignment,
)

__all__ = [
    "AMBIGUOUS_MARGIN",
    "DEFAULT_MAX_EVENTS",
    "DEFAULT_TOLERANCE_S",
    "INDISTINCT_GRAY_LEVELS",
    "AlignmentVerifier",
    "BoundaryCheck",
    "DatasetAudit",
    "EpisodeTrend",
    "LeRobotPTSAuditError",
    "PTSQuery",
    "VideoAudit",
    "VideoEvidence",
    "audit_lerobot_dataset",
    "classify_video",
    "float32_ulp_ns",
    "legacy_float32_residual_ns",
    "precision_band_ns",
    "render_audit_text",
    "HTTPAlignmentVerifier",
    "AudioClip",
    "BaseProvider",
    "Candidate",
    "ChannelReading",
    "EventEvidence",
    "EvidenceProvider",
    "FrameImage",
    "Inspection",
    "NullProvider",
    "PageContext",
    "StreamInfo",
    "VerificationRequest",
    "VerificationResult",
    "VerificationReview",
    "VerificationServiceError",
    "apply_verification_review",
    "build_page",
    "collect_evidence",
    "inspection_from_alignment",
    "perturb",
    "residuals_ns",
    "restrict_to_overlap",
    "review_verification",
    "verification_document",
    "verify_alignment",
]
