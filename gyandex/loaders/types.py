from typing import Any, Dict, Optional

from pydantic import BaseModel


class Document(BaseModel):
    title: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    content: str
