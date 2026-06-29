import tempfile
from pathlib import Path

from PIL import Image

from lorakit.datasets import DreamBoothDataset


def test_loads_sidecar_captions():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        img = Image.new("RGB", (64, 64))
        img.save(tmp / "test.png")
        (tmp / "test.txt").write_text("cat, hat, mat", encoding="utf-8")

        ds = DreamBoothDataset(
            str(tmp),
            instance_prompt="fallback",
            class_prompt="thing",
            resolution=64,
            caption_extension=".txt",
        )
        assert ds.custom_instance_prompts == ["cat, hat, mat"], (
            "custom_instance_prompts should load the sidecar caption"
        )
        assert ds[0]["instance_prompt"] == "cat, hat, mat", (
            "__getitem__ should return the sidecar caption as instance_prompt"
        )


def test_falls_back_to_instance_prompt_when_no_sidecar():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        img = Image.new("RGB", (64, 64))
        img.save(tmp / "test.png")

        ds = DreamBoothDataset(
            str(tmp),
            instance_prompt="fallback",
            class_prompt="thing",
            resolution=64,
            caption_extension=".txt",
        )
        assert ds.custom_instance_prompts is None, (
            "custom_instance_prompts should be None when no sidecars exist"
        )
        assert ds[0]["instance_prompt"] == "fallback", (
            "__getitem__ should fall back to instance_prompt"
        )


def test_empty_sidecar_falls_back_to_instance_prompt():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        img = Image.new("RGB", (64, 64))
        img.save(tmp / "test.png")
        (tmp / "test.txt").write_text("", encoding="utf-8")

        ds = DreamBoothDataset(
            str(tmp),
            instance_prompt="fallback",
            class_prompt="thing",
            resolution=64,
            caption_extension=".txt",
        )
        assert ds.custom_instance_prompts is None, (
            "empty sidecar should not populate custom_instance_prompts"
        )
        assert ds[0]["instance_prompt"] == "fallback", (
            "__getitem__ should fall back when caption is empty"
        )


def test_no_caption_extension_disables_feature():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        img = Image.new("RGB", (64, 64))
        img.save(tmp / "test.png")
        (tmp / "test.txt").write_text("cat, hat, mat", encoding="utf-8")

        ds = DreamBoothDataset(
            str(tmp),
            instance_prompt="fallback",
            class_prompt="thing",
            resolution=64,
        )
        assert ds.custom_instance_prompts is None, (
            "custom_instance_prompts should be None when caption_extension is not set"
        )
        assert ds[0]["instance_prompt"] == "fallback", (
            "__getitem__ should use instance_prompt when caption_extension is not set"
        )


def test_repeats_apply_to_captions():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        img = Image.new("RGB", (64, 64))
        img.save(tmp / "test.png")
        (tmp / "test.txt").write_text("cat, hat, mat", encoding="utf-8")

        ds = DreamBoothDataset(
            str(tmp),
            instance_prompt="fallback",
            class_prompt="thing",
            resolution=64,
            repeats=3,
            caption_extension=".txt",
        )
        assert ds.num_instance_images == 3, "1 image × 3 repeats = 3"
        assert len(ds.custom_instance_prompts) == 3, (
            "captions should be repeated to match images"
        )
        assert ds[0]["instance_prompt"] == "cat, hat, mat"
        assert ds[1]["instance_prompt"] == "cat, hat, mat"
        assert ds[2]["instance_prompt"] == "cat, hat, mat"
