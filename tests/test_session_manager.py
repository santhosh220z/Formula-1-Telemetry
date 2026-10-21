import pandas as pd
import pytest


class FakeSession:
    def __init__(self, laps, results=None):
        self.laps = laps
        self.results = results if results is not None else pd.DataFrame()


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


def test_has_timing_data_true_when_any_lap_timed():
    from core.session_manager import has_timing_data

    laps = pd.DataFrame({"LapTime": pd.to_timedelta([None, "00:01:31.5"])})
    assert has_timing_data(FakeSession(laps)) is True


def test_has_timing_data_false_when_all_laptimes_nat():
    from core.session_manager import has_timing_data

    laps = pd.DataFrame({"LapTime": pd.to_timedelta([None, None])})
    assert has_timing_data(FakeSession(laps)) is False


def test_has_timing_data_false_when_no_laps():
    from core.session_manager import has_timing_data

    laps = pd.DataFrame({"LapTime": pd.to_timedelta([])})
    assert has_timing_data(FakeSession(laps)) is False


def test_parse_lap_label_fastest_returns_none():
    from core.session_manager import parse_lap_label

    assert parse_lap_label("Fastest (default)") is None


def test_parse_lap_label_extracts_number():
    from core.session_manager import parse_lap_label

    assert parse_lap_label("Lap 5 · 1:20.3") == 5
    assert parse_lap_label("Lap 12") == 12


def test_parse_lap_label_rejects_garbage():
    from core.session_manager import parse_lap_label

    with pytest.raises(ValueError):
        parse_lap_label("nonsense")


def test_available_events_returns_filtered_schedule(monkeypatch):
    from core import session_manager

    schedule = pd.DataFrame(
        {
            "EventName": ["Pre-Season Testing 2024", "Bahrain Grand Prix"],
            "EventDate": pd.to_datetime(["2024-02-21", "2024-03-02"]),
        }
    )
    monkeypatch.setattr(
        session_manager.fastf1, "get_event_schedule", lambda year: schedule
    )

    assert session_manager.available_events(2024, timeout=5) == ["Bahrain Grand Prix"]


def test_available_events_times_out_on_stalled_fetch(monkeypatch):
    import time as _time

    from core import session_manager

    def stall(year):
        _time.sleep(5)

    monkeypatch.setattr(session_manager.fastf1, "get_event_schedule", stall)

    with pytest.raises(TimeoutError):
        session_manager.available_events(2024, timeout=0.2)


def test_available_events_reraises_fetch_failure(monkeypatch):
    from core import session_manager

    def boom(year):
        raise RuntimeError("api down")

    monkeypatch.setattr(session_manager.fastf1, "get_event_schedule", boom)

    with pytest.raises(RuntimeError, match="api down"):
        session_manager.available_events(2024, timeout=2)


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


def test_driver_candidates_prefers_results_metadata(monkeypatch):
    from core import session_manager

    monkeypatch.setattr(
        session_manager.fastf1.plotting, "get_driver_color", lambda *args, **kwargs: "#123456"
    )

    laps = pd.DataFrame({"Driver": ["VER"], "Team": ["Old Team"]})
    results = pd.DataFrame(
        {
            "Abbreviation": ["VER"],
            "FullName": ["Max Verstappen"],
            "TeamName": ["Red Bull Racing"],
        }
    )

    candidates = session_manager.driver_candidates(FakeSession(laps, results))

    assert candidates.loc["VER", "name"] == "Max Verstappen"
    assert candidates.loc["VER", "team"] == "Red Bull Racing"


def test_driver_candidates_falls_back_to_laps_team(monkeypatch):
    from core import session_manager

    monkeypatch.setattr(
        session_manager.fastf1.plotting, "get_driver_color", lambda *args, **kwargs: "#123456"
    )

    laps = pd.DataFrame({"Driver": ["VER"], "Team": ["Red Bull Racing"]})

    candidates = session_manager.driver_candidates(FakeSession(laps))

    assert candidates.loc["VER", "name"] == "VER"
    assert candidates.loc["VER", "team"] == "Red Bull Racing"


def test_driver_image_path_matches_exact_name(tmp_path, monkeypatch):
    from core import session_manager

    monkeypatch.setattr(session_manager, "DRIVER_IMAGES_DIR", str(tmp_path))
    folder = tmp_path / "2024"
    folder.mkdir()
    image = folder / "MAX VERSTAPPEN.png"
    image.write_bytes(b"png")

    assert session_manager.driver_image_path(2024, "Max Verstappen") == str(image)


def test_driver_image_path_normalizes_accents(tmp_path, monkeypatch):
    from core import session_manager

    monkeypatch.setattr(session_manager, "DRIVER_IMAGES_DIR", str(tmp_path))
    folder = tmp_path / "2024"
    folder.mkdir()
    image = folder / "NICO HULKENBERG.png"
    image.write_bytes(b"png")

    assert session_manager.driver_image_path(2024, "Nico Hülkenberg") == str(image)


def test_driver_image_path_maps_known_aliases(tmp_path, monkeypatch):
    from core import session_manager

    monkeypatch.setattr(session_manager, "DRIVER_IMAGES_DIR", str(tmp_path))
    folder = tmp_path / "2020"
    folder.mkdir()
    files = {
        "LEWIS HAMLITON.png": "Lewis Hamilton",
        "VALTARI BOTTAS.png": "Valtteri Bottas",
        "JACK AITKIN.png": "Jack Aitken",
        "ZHOU GUANYU.png": "Guanyu Zhou",
        "KIMI ANTONELLI.png": "Andrea Kimi Antonelli",
        "ALEXANDER ALBON.png": "Alex Albon",
        "CARLOS SAINZ.png": "Carlos Sainz Jr.",
    }
    for filename in files:
        (folder / filename).write_bytes(b"png")

    for filename, full_name in files.items():
        assert session_manager.driver_image_path(2020, full_name) == str(folder / filename)


def test_driver_image_path_prefers_exact_match_before_alias(tmp_path, monkeypatch):
    from core import session_manager

    monkeypatch.setattr(session_manager, "DRIVER_IMAGES_DIR", str(tmp_path))
    folder_2020 = tmp_path / "2020"
    folder_2021 = tmp_path / "2021"
    folder_2020.mkdir()
    folder_2021.mkdir()
    exact = folder_2020 / "DANIIL KVYAT.png"
    alias = folder_2021 / "DANIIL KYVAT.png"
    exact.write_bytes(b"png")
    alias.write_bytes(b"png")

    assert session_manager.driver_image_path(2020, "Daniil Kvyat") == str(exact)
    assert session_manager.driver_image_path(2021, "Daniil Kvyat") == str(alias)


def test_driver_image_path_returns_none_when_missing(tmp_path, monkeypatch):
    from core import session_manager

    monkeypatch.setattr(session_manager, "DRIVER_IMAGES_DIR", str(tmp_path))
    (tmp_path / "2024").mkdir()

    assert session_manager.driver_image_path(2024, "Max Verstappen") is None
    assert session_manager.driver_image_path(2019, "Max Verstappen") is None
