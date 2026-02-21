from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

SUFFICIENT = "SUFFICIENT"
INSUFFICIENT = "INSUFFICIENT"

CheckerResult = Tuple[str, float, Dict[str, Any]]


class BaseChecker(ABC):
    """문맥 충분성 체크 공통 인터페이스."""

    name: str = "base"

    @abstractmethod
    def predict(self, question: str, contexts: List[str]) -> CheckerResult:
        """(label, score, meta)를 반환한다."""
        raise NotImplementedError
