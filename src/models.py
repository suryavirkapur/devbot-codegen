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

class FileUpdate(BaseModel):
    path: str = Field(description="The path to the file that needs to be updated.")
    updatedCode: str = Field(description="The full, updated content of the file.")

class DebuggingUpdate(BaseModel):
    updates: List[FileUpdate] = Field(description="A list of file updates to fix the code.")

class DebugAttempt(BaseModel):
    attempt_number: int = Field(description="The attempt number (1-based)")
    error_logs: str = Field(description="The error logs from this attempt")
    strategy_used: str = Field(description="The debugging strategy that was attempted")
    files_modified: List[str] = Field(description="List of file paths that were modified in this attempt")
    success: bool = Field(description="Whether this attempt was successful")

class DebugSession(BaseModel):
    project_path: str = Field(description="Path to the project being debugged")
    attempts: List[DebugAttempt] = Field(default_factory=list, description="History of debug attempts")
    current_attempt: int = Field(default=0, description="Current attempt number")
    max_attempts: int = Field(default=5, description="Maximum number of attempts allowed")

class ValidationTest(BaseModel):
    test_type: str = Field(description="Type of validation test (e.g., 'endpoint_test', 'functionality_test', 'integration_test')")
    test_command: str = Field(description="Command to run the test")
    expected_output: Optional[str] = Field(default=None, description="Expected output pattern or response")
    timeout: int = Field(default=30, description="Timeout in seconds for the test")

class ValidationPlan(BaseModel):
    tests: List[ValidationTest] = Field(description="List of validation tests to run")
    setup_commands: List[str] = Field(default_factory=list, description="Commands to run before tests (e.g., database setup)")
    cleanup_commands: List[str] = Field(default_factory=list, description="Commands to run after tests")

class ValidationResult(BaseModel):
    test_type: str = Field(description="Type of test that was run")
    success: bool = Field(description="Whether the test passed")
    output: str = Field(description="Output from the test")
    error_message: Optional[str] = Field(default=None, description="Error message if test failed")

class ValidationReport(BaseModel):
    overall_success: bool = Field(description="Whether all validations passed")
    results: List[ValidationResult] = Field(description="Individual test results")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations for improvement")

class AIAgentCall(BaseModel):
    agent_type: str = Field(description="Type of AI agent (e.g., 'debugger', 'validator', 'optimizer')")
    prompt: str = Field(description="Prompt sent to the agent")
    response: str = Field(description="Response from the agent")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    model_used: str = Field(description="AI model used for this call")

class MultiAgentSession(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    project_context: str = Field(description="Context about the project being worked on")
    agent_calls: List[AIAgentCall] = Field(default_factory=list, description="History of AI agent calls")
    current_phase: str = Field(default="generation", description="Current phase: generation, debugging, validation, optimization")
