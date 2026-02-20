"""Test script for state color toggling."""

from visualizer.colors import state_color, CELL_STATES, COLORS


class MockCol:
    bursting = True


class MockCell:
    def __init__(self, active=False, predictive=False, prev_predictive=False, winner=False):
        self.active = active
        self.predictive = predictive
        self.prev_predictive = prev_predictive
        self.winner = winner


def test_state_colors():
    print("Testing state priority with hidden states:")
    print(f"Available states: {CELL_STATES}")

    # Test correct_prediction (needs predictive + prev_predictive + active)
    cell = MockCell(active=True, predictive=True, prev_predictive=True)
    assert state_color(cell) == COLORS["correct_prediction"], "correct_prediction visible failed"
    assert state_color(cell, hidden_states={"correct_prediction"}) == COLORS["predictive"], "correct_prediction hidden failed"
    print("✓ correct_prediction toggle works")

    # Test bursting
    cell = MockCell(active=True)
    col = MockCol()
    assert state_color(cell, col) == COLORS["bursting"], "bursting visible failed"
    assert state_color(cell, col, hidden_states={"bursting"}) == COLORS["active"], "bursting hidden failed"
    print("✓ bursting toggle works")

    # Test predictive
    cell = MockCell(predictive=True)
    assert state_color(cell) == COLORS["predictive"], "predictive visible failed"
    assert state_color(cell, hidden_states={"predictive"}) == COLORS["inactive"], "predictive hidden failed"
    print("✓ predictive toggle works")

    # Test winner
    cell = MockCell(winner=True)
    assert state_color(cell) == COLORS["winner"], "winner visible failed"
    assert state_color(cell, hidden_states={"winner"}) == COLORS["inactive"], "winner hidden failed"
    print("✓ winner toggle works")

    # Test active
    cell = MockCell(active=True)
    assert state_color(cell) == COLORS["active"], "active visible failed"
    assert state_color(cell, hidden_states={"active"}) == COLORS["inactive"], "active hidden failed"
    print("✓ active toggle works")

    # Test multiple hidden states
    cell = MockCell(active=True, predictive=True, prev_predictive=True, winner=True)
    # With all states hidden, should show inactive
    all_hidden = {"active", "predictive", "bursting", "winner", "correct_prediction"}
    assert state_color(cell, hidden_states=all_hidden) == COLORS["inactive"], "all hidden failed"
    print("✓ all states hidden falls back to inactive")

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_state_colors()
