from pydantic import BaseModel, Field, field_validator, model_validator
from PIL import Image as PILImage
import os
from pathlib import Path, WindowsPath, PosixPath
import json

# AgentEngine imports
from agent_engine.agent_logger import AgentLogger
logger = AgentLogger(__name__)

class Page(BaseModel):
    filename: str = Field(..., description="Page filename")
    image: PILImage.Image = Field(..., description="Page image")
    number: int = Field(..., description="Page number")

    @field_validator('filename', mode='before')
    @classmethod
    def validate_and_convert_path(cls, v) -> str:
        if v is None:
            return ""
        if isinstance(v, str):
            return v
        if isinstance(v, (Path, WindowsPath, PosixPath)):
            return str(v)
        try:
            return str(v)
        except Exception as e:
            logger.warning(f"Failed to convert path {v} to string: {e}")
            return ""

    def save(self):
        parent_dir = os.path.dirname(self.filename)
        os.makedirs(parent_dir, exist_ok=True)
        try:
            self.image.save(self.filename, format="PNG")
        except Exception as e:
            logger.error(f"Failed to save image: {e}")

        try:
            metadata_dir = os.path.join(parent_dir, f"page_{self.number}.json")
            metadata = {
                "number": self.number,
            }
            with open(metadata_dir, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=4)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")

    @classmethod
    def from_filename(cls, filename: str) -> "Page":
        image = PILImage.open(filename)
        number = 0
        metadata_dir = os.path.join(os.path.dirname(filename), f"page_{number}.json")
        if os.path.exists(metadata_dir):
            with open(metadata_dir, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                number = metadata.get("number", 0)

        return cls(filename=filename, image=image, number=number)

    class Config:
        # Allow creating objects from dictionaries
        from_attributes = True
        # Allow extra fields
        extra = "ignore"
        # Use aliases
        populate_by_name = True
        # Allow arbitrary types
        arbitrary_types_allowed = True