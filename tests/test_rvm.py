import numpy as np
import pytest
from sparce import SparseBayesianPairRanker, RVMConfig, CorrectivePolicy
from sparce.evaluation import evaluate_pairs, gate_candidate_update


def fixture_model():
    rng = np.random.default_rng(1)
    n = 140
    labels = np.r_[np.ones(n // 2), np.zeros(n // 2)].astype(int)
    a = np.r_[np.tile([[1., 0.]], (n//2, 1)), np.tile([[-1., 0.]], (n//2, 1))]
    b = -a
    a += rng.normal(0, 0.05, a.shape)
    b += rng.normal(0, 0.05, b.shape)
    z = np.array([[1, 0], [0.99, 0.1], [-1, 0], [-0.99, 0.1], [0, 1]], float)
    model = SparseBayesianPairRanker(RVMConfig(max_basis=5)).fit(z, a, b, np.full(n, -0.5), labels)
    return model, a, b, labels


def test_rvm_predict_and_snapshot():
    model, a, b, _ = fixture_model()
    assert 1 <= model.active_count <= 5
    pred = model.predict(a[:3], b[:3], [-0.5]*3)
    assert all(np.isfinite(v.corrected_margin) and v.correction_sd >= 0 for v in pred)
    reconstructed = SparseBayesianPairRanker.from_dict(model.to_dict())
    pred2 = reconstructed.predict(a[:3], b[:3], [-0.5]*3)
    assert np.allclose([p.corrected_margin for p in pred], [p.corrected_margin for p in pred2])


def test_scope_and_nonautomatic_application():
    model, a, b, _ = fixture_model()
    policy = CorrectivePolicy(model, {'version': '2.0'}, z_score=0.5)
    ok = policy.decide(a[0], b[0], -0.5, {'version':'2.0'})
    bad = policy.decide(a[0], b[0], -0.5, {'version':'1.0'}, independent_verification=True)
    assert ok.recommend_alternative
    assert not ok.auto_apply
    assert not bad.recommend_alternative
    assert bad.reason == 'scope_mismatch'
    verified = policy.decide(a[0], b[0], -0.5, {'version':'2.0'}, independent_verification=True)
    assert verified.auto_apply


def test_label_blind_eval_and_gate():
    model, a, b, labels = fixture_model()
    policy = CorrectivePolicy(model, {'version':'2.0'}, z_score=0.5)
    contexts = [{'version':'2.0'}]*len(labels)
    result = evaluate_pairs(policy, a, b, [-0.5]*len(labels), contexts, labels)
    assert result.corrected_correct > result.baseline_correct
    assert result.fixes > 0
    assert gate_candidate_update(result, result, min_gain=0.0, max_regressions=0)


def test_validation():
    with pytest.raises(ValueError):
        SparseBayesianPairRanker().fit([[1,0]], [[1,0]], [[0,1]], [-0.5], [1])
    with pytest.raises(ValueError):
        SparseBayesianPairRanker().fit([[1,0]], [[1,0],[0,1]], [[0,1],[1,0]], [-0.5]*2, [1,1])
