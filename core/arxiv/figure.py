from pydantic import BaseModel, Field, field_validator, model_validator
from PIL import Image as PILImage
import json
import os
from pathlib import Path, WindowsPath, PosixPath

# AgentEngine imports
from agent_engine.agent_logger import AgentLogger
logger = AgentLogger(__name__)

class Figure(BaseModel):
    filename: str = Field(..., description="Figure filename")
    image: PILImage.Image = Field(..., description="Figure image")
    number: int = Field(..., description="Figure number")
    caption: str = Field("", description="Figure caption")
    description: str = Field("", description="Figure description")

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
            metadata_dir = os.path.join(parent_dir, f"figure_{self.number}.json")
            metadata = {
                "number": self.number,
                "caption": self.caption,
                "description": self.description,
            }
            with open(metadata_dir, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=4)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")

    @classmethod
    def from_filename(cls, filename: str) -> "Figure":
        image = PILImage.open(filename)
        number = 0
        caption = ""
        description = ""
        metadata_dir = os.path.join(os.path.dirname(filename), f"figure_{number}.json")
        if os.path.exists(metadata_dir):
            with open(metadata_dir, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                number = metadata.get("number", 0)
                caption = metadata.get("caption", "")
                description = metadata.get("description", "")

        return cls(filename=filename, image=image, number=number, caption=caption, description=description)

    class Config:
        # Allow creating objects from dictionaries
        from_attributes = True
        # Allow extra fields
        extra = "ignore"
        # Use aliases
        populate_by_name = True
        # Allow arbitrary types
        arbitrary_types_allowed = True
    