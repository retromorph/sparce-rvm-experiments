import numpy as np
from sparce import SparseBayesianPairRanker, RVMConfig, CorrectivePolicy, SnapshotLedger


def test_non_destructive_rollback(tmp_path):
    model = SparseBayesianPairRanker(RVMConfig(max_basis=2, max_ard_iterations=3)).fit(
        [[1,0],[-1,0]], [[1,0], [-1,0]], [[-1,0], [1,0]], [-0.4,-0.4], [1,0])
    ledger = SnapshotLedger(tmp_path / 'ledger.sqlite')
    a = CorrectivePolicy(model, {'version':'2'})
    b = CorrectivePolicy(model, {'version':'3'})
    v1 = ledger.commit(a, reason='shadow_passed', evidence_ids=['proof-1'])
    v2 = ledger.commit(b, reason='validated_edit', evidence_ids=['proof-2'])
    assert v2 > v1
    v3 = ledger.rollback(v1)
    assert v3 > v2
    _, active = ledger.load()
    assert active.scope == {'version':'2'}
    assert [e['type'] for e in ledger.events()] == ['commit','commit','commit']
    assert len(ledger.events()) == 3
    ledger.close()
