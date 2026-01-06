from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class Example:
    key: str
    image_path: str
    src_prompt: str
    tar_prompt: str
    blended_word: Optional[Any] = None

    @staticmethod
    def from_dict(key: str, d: Dict[str, Any]) -> "Example":
        for k in ("image_path", "src_prompt", "tgt_prompt"):
            if k not in d:
                raise ValueError(
                    f"[dataset.json] item '{key}' missing field: {k}")

        return Example(
            key=key,
            image_path=d["image_path"],
            src_prompt=d["src_prompt"],
            tar_prompt=d["tgt_prompt"],
            blended_word=d.get("blended_word", None),
        )
