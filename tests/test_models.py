import pytest
from pydantic import ValidationError
from datetime import datetime
from uuid import UUID
import json

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import (
    CoreFeature, DataModel, TechnologyStack, BRDBase, BRD, BRDCreatePayload,
    BRDUpdatePayload, BRDTextCreatePayload, FileDependencyInfo, DependencyOrder,
    FileUpdate, DebuggingUpdate, DebugAttempt, DebugSession, ValidationTest,
    ValidationPlan, ValidationResult, ValidationReport, AIAgentCall,
    MultiAgentSession, UnifiedProjectRequest, ProjectGenerationResult,
    ProjectGenerationError
)


class TestCoreFeature:
    def test_valid_core_feature(self):
        feature = CoreFeature(
            name="User Authentication",
            description="Implement user login and registration",
            priority="High"
        )
        assert feature.name == "User Authentication"
        assert feature.priority == "High"
        assert isinstance(feature.id, UUID)
    
    def test_invalid_priority(self):
        with pytest.raises(ValidationError):
            CoreFeature(
                name="Test Feature",
                description="Test description",
                priority="Critical"  # Invalid priority
            )
    
    def test_empty_name(self):
        with pytest.raises(ValidationError):
            CoreFeature(
                name="",
                description="Test description",
                priority="High"
            )


class TestTechnologyStack:
    def test_empty_technology_stack(self):
        stack = TechnologyStack()
        assert stack.frontend == []
        assert stack.backend == []
        assert stack.database == []
        assert stack.other == []
    
    def test_populated_technology_stack(self):
        stack = TechnologyStack(
            frontend=["React", "TypeScript"],
            backend=["FastAPI", "Python"],
            database=["PostgreSQL"],
            other=["Docker", "Redis"]
        )
        assert len(stack.frontend) == 2
        assert "React" in stack.frontend
        assert "FastAPI" in stack.backend


class TestBRDCreatePayload:
    def test_minimal_brd_payload(self):
        brd = BRDCreatePayload(projectName="Test Project")
        assert brd.projectName == "Test Project"
        assert brd.projectDescription == ""
        assert isinstance(brd.technologyStack, TechnologyStack)
        assert brd.coreFeatures == []
    
    def test_full_brd_payload(self):
        features = [
            CoreFeature(
                name="User Auth",
                description="Authentication system",
                priority="High"
            )
        ]
        
        tech_stack = TechnologyStack(
            frontend=["React"],
            backend=["FastAPI"]
        )
        
        brd = BRDCreatePayload(
            projectName="Full Test Project",
            projectDescription="A comprehensive test project",
            technologyStack=tech_stack,
            coreFeatures=features,
            authentication="jwt",
            apiRequirements=["POST /api/auth/login", "GET /api/users"]
        )
        
        assert brd.projectName == "Full Test Project"
        assert len(brd.coreFeatures) == 1
        assert brd.authentication == "jwt"
        assert len(brd.apiRequirements) == 2
    
    def test_json_parsing_technology_stack(self):
        """Test that technology stack can be parsed from JSON string"""
        brd_data = {
            "projectName": "JSON Test",
            "technologyStack": '{"frontend": ["React"], "backend": ["FastAPI"]}'
        }
        
        brd = BRDCreatePayload.model_validate(brd_data)
        assert brd.technologyStack.frontend == ["React"]
        assert brd.technologyStack.backend == ["FastAPI"]


class TestUnifiedProjectRequest:
    def test_valid_unified_request(self):
        request = UnifiedProjectRequest(
            businessRequirement="I need a todo app with user authentication"
        )
        assert len(request.businessRequirement) > 10
        assert request.projectName is None
        assert request.additionalInstructions is None
    
    def test_unified_request_with_overrides(self):
        request = UnifiedProjectRequest(
            businessRequirement="Build an e-commerce platform with payment integration",
            projectName="My E-commerce App",
            additionalInstructions="Use Stripe for payments and implement real-time notifications"
        )
        assert request.projectName == "My E-commerce App"
        assert "Stripe" in request.additionalInstructions
    
    def test_too_short_business_requirement(self):
        with pytest.raises(ValidationError):
            UnifiedProjectRequest(businessRequirement="Too short")


class TestProjectGenerationResult:
    def test_successful_generation_result(self):
        brd_data = BRDCreatePayload(projectName="Test Project")
        
        result = ProjectGenerationResult(
            success=True,
            project_name="Test Project",
            brd_data=brd_data,
            zip_file_path="/path/to/project.zip",
            generation_time_seconds=45.5,
            total_files_generated=15,
            ai_agent_calls=8
        )
        
        assert result.success is True
        assert result.project_name == "Test Project"
        assert result.generation_time_seconds == 45.5
        assert result.total_files_generated == 15
        assert result.debug_attempts == 0  # Default value


class TestValidationModels:
    def test_validation_test(self):
        test = ValidationTest(
            test_type="endpoint_test",
            test_command="curl http://localhost:8080/health",
            expected_output="ok",
            timeout=30
        )
        assert test.test_type == "endpoint_test"
        assert test.timeout == 30
    
    def test_validation_plan(self):
        tests = [
            ValidationTest(
                test_type="build_test",
                test_command="docker build .",
                timeout=120
            )
        ]
        
        plan = ValidationPlan(
            tests=tests,
            setup_commands=["npm install"],
            cleanup_commands=["docker system prune -f"]
        )
        
        assert len(plan.tests) == 1
        assert "npm install" in plan.setup_commands
    
    def test_validation_result(self):
        result = ValidationResult(
            test_type="unit_test",
            success=True,
            output="All tests passed",
            error_message=None
        )
        assert result.success is True
        assert result.error_message is None
    
    def test_validation_report(self):
        results = [
            ValidationResult(
                test_type="build_test",
                success=True,
                output="Build successful"
            ),
            ValidationResult(
                test_type="unit_test",
                success=False,
                output="Test failed",
                error_message="TypeError: Cannot read property"
            )
        ]
        
        report = ValidationReport(
            overall_success=False,
            results=results,
            recommendations=["Fix the TypeError in unit tests", "Add error handling"]
        )
        
        assert report.overall_success is False
        assert len(report.results) == 2
        assert len(report.recommendations) == 2


class TestFileDependencyInfo:
    def test_file_dependency_info(self):
        file_info = FileDependencyInfo(
            path="src/main.py",
            description="Main application entry point",
            dependsOn=["src/config.py", "src/models.py"]
        )
        
        assert file_info.path == "src/main.py"
        assert len(file_info.dependsOn) == 2
        assert "src/config.py" in file_info.dependsOn
    
    def test_no_dependencies(self):
        file_info = FileDependencyInfo(
            path="README.md",
            description="Project documentation"
        )
        
        assert file_info.dependsOn == []


class TestMultiAgentSession:
    def test_multi_agent_session(self):
        session = MultiAgentSession(
            project_context="Test project generation"
        )
        
        assert isinstance(session.session_id, str)
        assert session.current_phase == "generation"
        assert session.agent_calls == []
    
    def test_agent_call_tracking(self):
        session = MultiAgentSession(
            project_context="Test project"
        )
        
        agent_call = AIAgentCall(
            agent_type="code_generator",
            prompt="Generate a FastAPI app",
            response="# FastAPI application code",
            model_used="gpt-4.1"
        )
        
        session.agent_calls.append(agent_call)
        assert len(session.agent_calls) == 1
        assert session.agent_calls[0].agent_type == "code_generator"


class TestDebugSession:
    def test_debug_session(self):
        debug_session = DebugSession(
            project_path="/path/to/project"
        )
        
        assert debug_session.current_attempt == 0
        assert debug_session.max_attempts == 5
        assert debug_session.attempts == []
    
    def test_debug_attempt(self):
        attempt = DebugAttempt(
            attempt_number=1,
            error_logs="Build failed: syntax error",
            strategy_used="Fix syntax errors",
            files_modified=["src/main.py"],
            success=False
        )
        
        assert attempt.attempt_number == 1
        assert attempt.success is False
        assert len(attempt.files_modified) == 1


if __name__ == "__main__":
    pytest.main([__file__]) 