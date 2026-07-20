from typing import Any

from pydantic import BaseModel


class SuccessResponse(BaseModel):
    """
    Standard success response.
    """

    success: bool = True
    data: Any