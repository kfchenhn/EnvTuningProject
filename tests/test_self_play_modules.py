from env_tuning.self_play.ast_diagnostics import ASTTrajectoryDiagnostic
from env_tuning.self_play.anchor_selector import AnchorSelector, TrajectoryRecord


def test_ast_first_divergence_parameter():
    diag = ASTTrajectoryDiagnostic()
    success = '[{"name":"book_hotel","arguments":{"city":"shanghai","days":2}}]'
    failed = '[{"name":"book_hotel","arguments":{"city":"beijing","days":2}}]'
    report = diag.build_counterfactual_report(success, failed)
    assert report["has_divergence"] is True
    assert report["divergence_type"] == "parameter"


def test_anchor_selector_priority():
    selector = AnchorSelector()
    peer_ok = TrajectoryRecord("task_a", "[]", 0.8, True, source="peer")
    peer_fail = TrajectoryRecord("task_a", "[]", 0.1, False, source="peer")
    anchor, source = selector.choose_anchor("task_a", [peer_fail, peer_ok])
    assert source == "peer"
    assert anchor is peer_ok

    selector = AnchorSelector()
    hist_ok = TrajectoryRecord("task_b", "[]", 0.7, True, source="history")
    selector.add_history(hist_ok)
    anchor, source = selector.choose_anchor("task_b", [peer_fail])
    assert source == "history"
    assert anchor is hist_ok
