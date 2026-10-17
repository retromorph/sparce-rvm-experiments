from sparce.contracts import Candidate, CodeRequest, VerificationOutcome, VerifiedPair
from sparce.stats import paired_accuracy_delta_interval, wilson_upper_onesided_95


def test_ambiguous_pair_refused():
    q = CodeRequest("id", "fix", "p", "2", "family", "2026-01-01", "sha256:x")
    a = Candidate("a", -1.0)
    b = Candidate("b", -0.1)
    verified = VerificationOutcome("pass", "tests", "hash1", 1.0)
    uncertain = VerificationOutcome("timeout", "tests", "hash2", 10.0)
    assert VerifiedPair(q, b, a, verified, verified).pair_label is None
    assert VerifiedPair(q, b, a, uncertain, verified).pair_label is None
    failed = VerificationOutcome("fail", "tests", "hash3", 1.0)
    assert VerifiedPair(q, b, a, failed, verified).pair_label == 1
    assert VerifiedPair(q, b, a, verified, failed).pair_label == 0


def test_wilson_upper():
    assert 0 < wilson_upper_onesided_95(0, 1000) < 0.01
    assert wilson_upper_onesided_95(100, 100) == 1.0


def test_paired_bootstrap_and_cluster():
    base = [True, False, False, True, False, False]
    treated = [True, True, True, True, False, True]
    point, lower, upper = paired_accuracy_delta_interval(
        base, treated, cluster_ids=["a", "a", "b", "b", "c", "c"],
        n_boot=200, seed=99,
    )
    assert point > 0
    assert lower <= point <= upper
