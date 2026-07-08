from mlxscribe.cli import LONG_FORM_THRESHOLD_SECONDS, normalize_language, resolve_context_mode


def test_normalize_language_maps_auto_to_none() -> None:
    assert normalize_language("auto") is None
    assert normalize_language("AUTO") is None


def test_normalize_language_preserves_explicit_language() -> None:
    assert normalize_language("ja") == "ja"


def test_resolve_context_auto_keeps_context_for_short_inputs() -> None:
    decision = resolve_context_mode(
        requested_mode="auto",
        duration_seconds=LONG_FORM_THRESHOLD_SECONDS,
    )

    assert decision["enabled"] is True
    assert decision["requested"] == "auto"


def test_resolve_context_auto_disables_context_for_long_inputs() -> None:
    decision = resolve_context_mode(
        requested_mode="auto",
        duration_seconds=LONG_FORM_THRESHOLD_SECONDS + 1,
    )

    assert decision["enabled"] is False
    assert decision["requested"] == "auto"


def test_resolve_context_explicit_modes_override_duration() -> None:
    assert resolve_context_mode(requested_mode="on", duration_seconds=999999)["enabled"] is True
    assert resolve_context_mode(requested_mode="off", duration_seconds=1)["enabled"] is False


def test_resolve_context_legacy_override_wins() -> None:
    decision = resolve_context_mode(
        requested_mode="auto",
        duration_seconds=1,
        explicit_override=False,
    )

    assert decision["enabled"] is False
    assert decision["reason"] == "explicit legacy override"
