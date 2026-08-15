from app.ingestion.dataset import (
    LANGUAGES,
    SPLITS,
    parquet_file,
    _lang_prefix,
)


def test_all_languages_have_prefixes():
    assert len(LANGUAGES) == 14
    assert LANGUAGES["hi"] == "hin"
    assert LANGUAGES["te"] == "tel"


def test_parquet_file_naming():
    assert parquet_file("hi", "train") == "train/hintrain.parquet"
    assert parquet_file("hi", "validation") == "validation/hinval.parquet"
    assert parquet_file("te", "validation") == "validation/telval.parquet"


def test_unknown_language_raises():
    try:
        _lang_prefix("xx")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown language")


def test_unknown_split_raises():
    try:
        parquet_file("hi", "test")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown split")


def test_splits():
    assert SPLITS == ("train", "validation")
