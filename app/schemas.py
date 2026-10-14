from pydantic import BaseModel, Field


class TextIndexRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1)


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    top_k: int = Field(default=4, ge=1, le=10)


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]