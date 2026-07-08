from mlxscribe.cli import normalize_language


def test_normalize_language_maps_auto_to_none() -> None:
    assert normalize_language("auto") is None
    assert normalize_language("AUTO") is None


def test_normalize_language_preserves_explicit_language() -> None:
    assert normalize_language("ja") == "ja"
