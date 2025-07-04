from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Literal, Dict, Any, Union
from uuid import UUID, uuid4
from datetime import datetime
import json

# Enums
AuthType = Literal["none", "basic", "oauth", "jwt"]
PriorityType = Literal["High", "Medium", "Low"]

# Enhanced Language and Framework Detection
class LanguageInfo(BaseModel):
    language: str = Field(description="Programming language (e.g., python, javascript, go, rust)")
    version: Optional[str] = Field(default=None, description="Language version if detected")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score of detection")

class FrameworkInfo(BaseModel):
    name: str = Field(description="Framework name (e.g., fastapi, express, gin, actix)")
    version: Optional[str] = Field(default=None, description="Framework version if detected")
    category: str = Field(description="Framework category (web, ml, mobile, etc.)")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score of detection")

class TechStackDetection(BaseModel):
    primary_language: LanguageInfo = Field(description="Primary programming language")
    secondary_languages: List[LanguageInfo] = Field(default_factory=list, description="Additional languages detected")
    frameworks: List[FrameworkInfo] = Field(default_factory=list, description="Detected frameworks")
    package_managers: List[str] = Field(default_factory=list, description="Package managers (npm, pip, cargo, etc.)")
    build_tools: List[str] = Field(default_factory=list, description="Build tools (webpack, vite, maven, etc.)")
    deployment_tools: List[str] = Field(default_factory=list, description="Deployment tools (docker, k8s, etc.)")
    databases: List[str] = Field(default_factory=list, description="Database technologies detected")

# Advanced Error Classification
class ErrorCategory(BaseModel):
    category: str = Field(description="Error category (syntax, dependency, config, runtime, etc.)")
    subcategory: Optional[str] = Field(default=None, description="Specific subcategory")
    severity: Literal["low", "medium", "high", "critical"] = Field(description="Error severity")
    confidence: float = Field(ge=0.0, le=1.0, description="Classification confidence")

class ErrorAnalysis(BaseModel):
    original_error: str = Field(description="Original error message")
    classification: ErrorCategory = Field(description="Error classification")
    affected_files: List[str] = Field(default_factory=list, description="Files likely affected by this error")
    suggested_fixes: List[str] = Field(default_factory=list, description="AI-suggested fixes")
    related_errors: List[str] = Field(default_factory=list, description="Related error patterns")
    root_cause: Optional[str] = Field(default=None, description="Potential root cause")

# Multi-Strategy Testing System
class TestStrategy(BaseModel):
    strategy_id: str = Field(description="Unique strategy identifier")
    name: str = Field(description="Human-readable strategy name")
    description: str = Field(description="Strategy description")
    applicable_languages: List[str] = Field(description="Languages this strategy applies to")
    applicable_frameworks: List[str] = Field(description="Frameworks this strategy applies to")
    priority: int = Field(description="Strategy priority (lower = higher priority)")
    estimated_time: int = Field(description="Estimated execution time in seconds")
    success_rate: float = Field(ge=0.0, le=1.0, description="Historical success rate")

class ParallelTestExecution(BaseModel):
    execution_id: str = Field(default_factory=lambda: str(uuid4()))
    strategies: List[TestStrategy] = Field(description="Strategies to execute in parallel")
    max_parallel: int = Field(default=3, description="Maximum parallel executions")
    timeout: int = Field(default=300, description="Total timeout in seconds")

class TestResult(BaseModel):
    strategy_id: str = Field(description="Strategy that produced this result")
    success: bool = Field(description="Whether the test passed")
    execution_time: float = Field(description="Execution time in seconds")
    output: str = Field(description="Test output")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    files_modified: List[str] = Field(default_factory=list, description="Files modified by this strategy")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence in the result")

# Progressive Learning System
class FixPattern(BaseModel):
    pattern_id: str = Field(default_factory=lambda: str(uuid4()))
    error_signature: str = Field(description="Error signature/pattern")
    language: str = Field(description="Programming language")
    framework: Optional[str] = Field(default=None, description="Framework if applicable")
    fix_description: str = Field(description="Description of the fix")
    success_count: int = Field(default=1, description="Number of times this fix worked")
    failure_count: int = Field(default=0, description="Number of times this fix failed")
    last_used: datetime = Field(default_factory=datetime.utcnow)
    code_changes: List[Dict[str, Any]] = Field(description="Specific code changes made")

class LearningSession(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    project_context: str = Field(description="Project context")
    patterns_learned: List[FixPattern] = Field(default_factory=list)
    patterns_applied: List[str] = Field(default_factory=list, description="Pattern IDs applied")
    success_patterns: List[str] = Field(default_factory=list, description="Successful pattern IDs")

# Comprehensive Testing Suite
class TestSuite(BaseModel):
    name: str = Field(description="Test suite name")
    test_type: Literal["unit", "integration", "performance", "security", "compatibility"] = Field(description="Type of test")
    commands: List[str] = Field(description="Commands to run")
    setup_commands: List[str] = Field(default_factory=list, description="Setup commands")
    cleanup_commands: List[str] = Field(default_factory=list, description="Cleanup commands")
    timeout: int = Field(default=60, description="Timeout in seconds")
    required_files: List[str] = Field(default_factory=list, description="Files that must exist")
    environment_vars: Dict[str, str] = Field(default_factory=dict, description="Required environment variables")

class ComprehensiveValidation(BaseModel):
    build_validation: TestSuite = Field(description="Build/compilation validation")
    unit_tests: Optional[TestSuite] = Field(default=None, description="Unit test suite")
    integration_tests: Optional[TestSuite] = Field(default=None, description="Integration test suite")
    performance_tests: Optional[TestSuite] = Field(default=None, description="Performance test suite")
    security_tests: Optional[TestSuite] = Field(default=None, description="Security test suite")
    compatibility_tests: Optional[TestSuite] = Field(default=None, description="Compatibility test suite")

# Smart Rollback System
class ProjectCheckpoint(BaseModel):
    checkpoint_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    description: str = Field(description="Checkpoint description")
    files_snapshot: Dict[str, str] = Field(description="File contents at this point")
    test_results: Optional[Dict[str, Any]] = Field(default=None, description="Test results at this checkpoint")
    working_state: bool = Field(description="Whether this was a working state")

class RollbackManager(BaseModel):
    project_path: str = Field(description="Project path")
    checkpoints: List[ProjectCheckpoint] = Field(default_factory=list)
    current_checkpoint: Optional[str] = Field(default=None, description="Current checkpoint ID")
    max_checkpoints: int = Field(default=10, description="Maximum checkpoints to keep")

# Enhanced Debug Session
class AdvancedDebugAttempt(BaseModel):
    attempt_number: int = Field(description="The attempt number (1-based)")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    tech_stack: TechStackDetection = Field(description="Detected technology stack")
    error_analysis: List[ErrorAnalysis] = Field(description="Analyzed errors")
    strategies_attempted: List[TestResult] = Field(description="All strategies attempted")
    best_result: Optional[TestResult] = Field(default=None, description="Best result from this attempt")
    checkpoint_created: Optional[str] = Field(default=None, description="Checkpoint ID if created")
    learning_applied: List[str] = Field(default_factory=list, description="Learning patterns applied")
    success: bool = Field(default=False, description="Whether this attempt was successful")

class AdvancedDebugSession(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    project_path: str = Field(description="Path to the project being debugged")
    tech_stack: Optional[TechStackDetection] = Field(default=None, description="Detected technology stack")
    attempts: List[AdvancedDebugAttempt] = Field(default_factory=list, description="History of debug attempts")
    current_attempt: int = Field(default=0, description="Current attempt number")
    max_attempts: int = Field(default=10, description="Maximum number of attempts allowed")
    rollback_manager: RollbackManager = Field(description="Rollback management")
    learning_session: LearningSession = Field(description="Learning session")
    success_patterns: List[str] = Field(default_factory=list, description="Successful patterns from this session")

# Quality Metrics
class CodeQualityMetrics(BaseModel):
    complexity_score: float = Field(ge=0.0, le=100.0, description="Code complexity score")
    maintainability_score: float = Field(ge=0.0, le=100.0, description="Maintainability score")
    test_coverage: float = Field(ge=0.0, le=100.0, description="Test coverage percentage")
    security_score: float = Field(ge=0.0, le=100.0, description="Security score")
    performance_score: float = Field(ge=0.0, le=100.0, description="Performance score")
    best_practices_compliance: float = Field(ge=0.0, le=100.0, description="Best practices compliance")

class ProjectHealth(BaseModel):
    overall_health: float = Field(ge=0.0, le=100.0, description="Overall project health score")
    quality_metrics: CodeQualityMetrics = Field(description="Code quality metrics")
    build_success: bool = Field(description="Whether project builds successfully")
    tests_passing: bool = Field(description="Whether tests are passing")
    security_issues: List[str] = Field(default_factory=list, description="Security issues found")
    performance_issues: List[str] = Field(default_factory=list, description="Performance issues found")
    recommendations: List[str] = Field(default_factory=list, description="Improvement recommendations")

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

class UnifiedProjectRequest(BaseModel):
    businessRequirement: str = Field(min_length=10, description="Business requirement as text or markdown")
    projectName: Optional[str] = Field(default=None, description="Optional project name override")
    additionalInstructions: Optional[str] = Field(default=None, description="Additional instructions for code generation")

class ProjectGenerationResult(BaseModel):
    success: bool = Field(description="Whether the project generation was successful")
    project_name: str = Field(description="Name of the generated project")
    brd_data: BRDCreatePayload = Field(description="Generated Business Requirements Document")
    zip_file_path: str = Field(description="Path to the generated zip file")
    generation_time_seconds: float = Field(description="Total time taken for generation")
    validation_report: Optional[ValidationReport] = Field(default=None, description="Validation test results")
    debug_attempts: int = Field(default=0, description="Number of debug attempts made")
    total_files_generated: int = Field(description="Total number of files generated")
    ai_agent_calls: int = Field(description="Total number of AI agent calls made")
    # Enhanced fields
    tech_stack_detected: Optional[TechStackDetection] = Field(default=None, description="Detected technology stack")
    project_health: Optional[ProjectHealth] = Field(default=None, description="Final project health assessment")
    advanced_debug_session: Optional[AdvancedDebugSession] = Field(default=None, description="Advanced debug session data")
    
class ProjectGenerationError(BaseModel):
    success: bool = Field(default=False)
    error_type: str = Field(description="Type of error that occurred")
    error_message: str = Field(description="Detailed error message")
    stage: str = Field(description="Stage where the error occurred")
    partial_results: Optional[Dict] = Field(default=None, description="Any partial results if available")
