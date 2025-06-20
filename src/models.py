from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Literal
from uuid import UUID, uuid4
from datetime import datetime
import json

# Enums
AuthType = Literal["none", "basic", "oauth", "jwt"]
PriorityType = Literal["High", "Medium", "Low"]

class CoreFeature(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    priority: PriorityType

class DataModel(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str = Field(min_length=1)
    fields: str
    relationships: str

class TechnologyStack(BaseModel):
    frontend: List[str] = Field(default_factory=list)
    backend: List[str] = Field(default_factory=list)
    database: List[str] = Field(default_factory=list)
    other: List[str] = Field(default_factory=list)

class BRDBase(BaseModel):
    projectName: str = Field(min_length=1)
    projectDescription: str = ""
    technologyStack: TechnologyStack = Field(default_factory=TechnologyStack)
    coreFeatures: List[CoreFeature] = []
    dataModels: List[DataModel] = []
    authentication: AuthType = "none"
    apiRequirements: List[str] = []
    additionalRequirements: Optional[str] = None

    @field_validator('technologyStack', mode='before')
    @classmethod
    def parse_technology_stack(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v

    @field_validator('coreFeatures', mode='before')
    @classmethod
    def parse_core_features(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v

    @field_validator('dataModels', mode='before')
    @classmethod
    def parse_data_models(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v

    @field_validator('apiRequirements', mode='before')
    @classmethod
    def parse_api_requirements(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v

    class Config:
        populate_by_name = True

class BRD(BRDBase):
    id: UUID = Field(default_factory=uuid4)
    createdAt: datetime = Field(default_factory=datetime.utcnow)
    updatedAt: datetime = Field(default_factory=datetime.utcnow)

class BRDCreatePayload(BRDBase):
    pass

class BRDUpdatePayload(BaseModel):
    projectName: Optional[str] = None
    projectDescription: Optional[str] = None
    technologyStack: Optional[TechnologyStack] = None
    coreFeatures: Optional[List[CoreFeature]] = None
    dataModels: Optional[List[DataModel]] = None
    authentication: Optional[AuthType] = None
    apiRequirements: Optional[List[str]] = None
    additionalRequirements: Optional[str] = None

    @field_validator('technologyStack', mode='before')
    @classmethod
    def parse_technology_stack(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v

    @field_validator('coreFeatures', mode='before')
    @classmethod
    def parse_core_features(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v

    @field_validator('dataModels', mode='before')
    @classmethod
    def parse_data_models(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v

    @field_validator('apiRequirements', mode='before')
    @classmethod
    def parse_api_requirements(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v


class BRDTextCreatePayload(BaseModel):
    businessInfo: str =  Field(min_length=5)

class FileDependencyInfo(BaseModel):
    path: str = Field(description="The full path of the file to be created, relative to the project root.")
    description: str = Field(description="A detailed description of the code or content that should be in this file.")
    dependsOn: List[str] = Field(default_factory=list, description="An array of file paths that this file depends on. Provide an empty array [] if there are no dependencies.")

class DependencyOrder(BaseModel):
    files: List[FileDependencyInfo] = Field(description="An array of files to be generated for the project, ordered by dependency.")
