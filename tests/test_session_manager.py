import pandas as pd
import pytest


class FakeSession:
    def __init__(self, laps):
        self.laps = laps


def test_session_codes_allowlist():
    from core.session_manager import SESSION_CODES

    assert SESSION_CODES == ["Q", "R", "FP1", "FP2", "FP3"]


def test_validate_session_code_returns_uppercase_code():
    from core.session_manager import validate_session_code

    assert validate_session_code("q") == "Q"
    assert validate_session_code("fp2") == "FP2"


def test_validate_session_code_rejects_unknown():
    from core.session_manager import validate_session_code

    with pytest.raises(ValueError):
        validate_session_code("X")


def test_filter_event_schedule_drops_testing_keeps_chronology():
    from core.session_manager import filter_event_schedule

    schedule = pd.DataFrame(
        {
            "EventName": [
                "Pre-Season Testing 2024",
                "Bahrain Grand Prix",
                "Saudi Arabian Grand Prix",
                "TESTING something",
            ],
            "EventDate": pd.to_datetime(
                ["2024-02-21", "2024-03-02", "2024-03-09", "2024-05-01"]
            ),
        }
    )

    events = filter_event_schedule(schedule)

    assert events == ["Bahrain Grand Prix", "Saudi Arabian Grand Prix"]


def test_driver_candidates_lists_codes_with_color_fallback(monkeypatch):
    from core import session_manager

    def boom(*args, **kwargs):
        raise RuntimeError("plotting API changed")

    monkeypatch.setattr(session_manager.fastf1.plotting, "get_driver_color", boom)

    laps = pd.DataFrame(
        {
            "Driver": ["VER", "ALB", "VER"],
            "FullName": ["Max Verstappen", "Alexander Albon", "Max Verstappen"],
            "TeamName": ["Red Bull", "Williams", "Red Bull"],
        }
    )

    candidates = session_manager.driver_candidates(FakeSession(laps))

    assert list(candidates.index) == ["ALB", "VER"]
    assert candidates.loc["VER", "name"] == "Max Verstappen"
    assert candidates.loc["VER", "team"] == "Red Bull"
    assert candidates["color"].notna().all()
    assert candidates["color"].nunique() == 2
