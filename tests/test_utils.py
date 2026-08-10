import pytest

from src.utils import load_config, save_json


def test_load_config_requires_mapping(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="non-empty YAML mapping"):
        load_config(path)


def test_load_config_and_save_json(tmp_path):
    config_path = tmp_path / "config.yaml"
    output_path = tmp_path / "nested" / "output.json"
    config_path.write_text("seed: 42\n", encoding="utf-8")

    config = load_config(config_path)
    save_json(config, output_path)

    assert config == {"seed": 42}
    assert output_path.is_file()
