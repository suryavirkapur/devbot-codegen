from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from uuid import UUID, uuid4
from datetime import datetime

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
    fields: str # Could be a structured dict or list in a more advanced version
    relationships: str # Could be a structured dict or list

class TechnologyStack(BaseModel):
    frontend: List[str] = Field(default_factory=list)
    backend: List[str] = Field(default_factory=list)
    database: List[str] = Field(default_factory=list)
    other: List[str] = Field(default_factory=list)

class BRDBase(BaseModel):
    projectName: str = Field(min_length=1, alias="projectName")
    projectDescription: str = Field(min_length=1, alias="projectDescription")
    technologyStack: TechnologyStack = Field(default_factory=TechnologyStack)
    coreFeatures: List[CoreFeature] = Field(default_factory=list)
    dataModels: List[DataModel] = Field(default_factory=list)
    authentication: AuthType = "none"
    apiRequirements: List[str] = Field(default_factory=list)
    additionalRequirements: Optional[str] = None

    class Config:
        populate_by_name = True # Allows using alias for field names

class BRD(BRDBase):
    id: UUID = Field(default_factory=uuid4)
    createdAt: datetime = Field(default_factory=datetime.utcnow)
    updatedAt: datetime = Field(default_factory=datetime.utcnow)

class BRDCreatePayload(BRDBase):
    pass

class BRDUpdatePayload(BRDBase):
    projectName: str = ""
    projectDescription: str = ""
    technologyStack: TechnologyStack = TechnologyStack()
    coreFeatures: List[CoreFeature] = []
    dataModels: List[DataModel] = []
    authentication: AuthType = "none"
    apiRequirements: List[str] = []

class BRDTextCreatePayload(BaseModel):
    businessInfo: str = Field(min_length=1)

# For brdGenerateRepo.ts DependencyOrderSchema
class FileDependencyInfo(BaseModel):
    path: str = Field(description="The full path of the file to be created, relative to the project root.")
    description: str = Field(description="A detailed description of the code or content that should be in this file.")
    dependsOn: List[str] = Field(default_factory=list, description="An array of file paths that this file depends on. Provide an empty array [] if there are no dependencies.")

class DependencyOrder(BaseModel):
    files: List[FileDependencyInfo] = Field(description="An array of files to be generated for the project, ordered by dependency.")
