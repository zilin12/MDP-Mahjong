from mahjong_robot import trials as T


def test_gate2_scoring_thresholds(tmp_path):
    for skill in T.GATE2_SKILLS:
        for i in range(50):
            ok = i >= (5 if skill == "insertion" else 4)  # insertion: 45/50 = 90%, others 46/50
            T.record(2, {"kind": skill, "success": ok, "reason": "" if ok else "slipped"}, tmp_path)
    for t in range(34):
        for i in range(10):
            T.record(2, {"kind": "recognition", "tile_type": t, "success": i != 0 or t != 5, "reason": "misread"}, tmp_path)
    rep = T.score_gate2(T.load(2, tmp_path))
    assert rep["skills"]["insertion"]["passed"] and rep["skills"]["insertion"]["successes"] == 45
    assert rep["recognition"]["passed"] and rep["passed"]


def test_gate2_needs_all_fifty_trials(tmp_path):
    for i in range(49):
        T.record(2, {"kind": "placement", "success": True}, tmp_path)
    assert not T.score_gate2(T.load(2, tmp_path))["skills"]["placement"]["passed"]


def test_gate4_corrective_touch_fails_a_hand(tmp_path):
    for i in range(5):
        T.record(4, {"kind": "hand", "success": True, "corrective_touch": i < 3}, tmp_path)
    rep = T.score_gate4(T.load(4, tmp_path))
    assert rep["hands"]["complete_without_touch"] == 2 and not rep["hands"]["passed"]


def test_failed_trial_requires_reason(tmp_path):
    try:
        T.record(4, {"kind": "pung", "success": False}, tmp_path)
    except ValueError:
        return
    raise AssertionError("expected a ValueError")
