import pytest

from A12.service.pipeline import validate_video


def test_validate_video_requires_input():
    with pytest.raises(ValueError, match="required"):
        validate_video(None)


def test_validate_video_rejects_unknown_extension(tmp_path):
    path = tmp_path / "not_a_video.txt"
    path.write_text("hello")
    with pytest.raises(ValueError, match="Unsupported"):
        validate_video(str(path))
