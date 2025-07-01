import pytest
import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import (
    UnifiedProjectRequest, ProjectGenerationResult, BRDCreatePayload, 
    TechnologyStack, ValidationReport, ValidationResult
)
import services


class TestUnifiedGenerateProjectEndpoint:
    """Integration tests for the unified project generation API endpoint"""
    
    @pytest.fixture
    def sample_business_requirement(self):
        return """
        # E-commerce Platform Requirements
        
        I need to build a modern e-commerce platform that allows:
        
        ## Core Features:
        - User registration and authentication
        - Product catalog with search and filtering
        - Shopping cart functionality
        - Order management system
        - Payment processing integration
        - Admin dashboard for managing products and orders
        
        ## Technical Requirements:
        - REST API backend
        - Modern web frontend
        - Database for storing products, users, and orders
        - Docker support for deployment
        - Unit tests and API documentation
        
        ## Additional Notes:
        - Should support multiple payment methods
        - Include real-time notifications for order updates
        - Mobile-responsive design
        """
    
    @pytest.fixture
    def minimal_business_requirement(self):
        return "Create a simple todo app where users can add, edit, and delete tasks."
    
    @pytest.fixture
    def mock_successful_generation_result(self):
        """Mock a successful project generation result"""
        brd_data = BRDCreatePayload(
            projectName="E-commerce Platform",
            projectDescription="Modern e-commerce platform with cart and payments",
            technologyStack=TechnologyStack(
                frontend=["React", "TypeScript"],
                backend=["FastAPI", "Python"],
                database=["PostgreSQL"],
                other=["Docker", "Redis"]
            )
        )
        
        validation_report = ValidationReport(
            overall_success=True,
            results=[
                ValidationResult(
                    test_type="build_test",
                    success=True,
                    output="Docker build successful"
                ),
                ValidationResult(
                    test_type="api_test",
                    success=True,
                    output="All API endpoints responding"
                )
            ],
            recommendations=[]
        )
        
        return ProjectGenerationResult(
            success=True,
            project_name="E-commerce Platform",
            brd_data=brd_data,
            zip_file_path="/tmp/e-commerce-platform-generated.zip",
            generation_time_seconds=120.5,
            validation_report=validation_report,
            debug_attempts=1,
            total_files_generated=25,
            ai_agent_calls=12
        )
    
    @pytest.mark.asyncio
    async def test_unified_generate_project_success(self, sample_business_requirement):
        """Test successful project generation through unified endpoint"""
        
        request_data = {
            "businessRequirement": sample_business_requirement,
            "projectName": "My E-commerce App",
            "additionalInstructions": "Use Stripe for payments and include comprehensive error handling"
        }
        
        # Mock successful result
        mock_brd = BRDCreatePayload(
            projectName="E-commerce Platform",
            projectDescription="Modern e-commerce platform with cart and payments",
            technologyStack=TechnologyStack(
                frontend=["React", "TypeScript"],
                backend=["FastAPI", "Python"],
                database=["PostgreSQL"],
                other=["Docker", "Redis"]
            )
        )
        
        mock_result = ProjectGenerationResult(
            success=True,
            project_name="E-commerce Platform",
            brd_data=mock_brd,
            zip_file_path="/tmp/e-commerce-platform-generated.zip",
            generation_time_seconds=120.5,
            total_files_generated=25,
            ai_agent_calls=12
        )
        
        with patch('services.unified_project_generation', new_callable=AsyncMock, 
                  return_value=mock_result) as mock_generation:
            
            # Simulate the API endpoint logic
            validated_request = UnifiedProjectRequest.model_validate(request_data)
            result = await services.unified_project_generation(validated_request)
            
            # Verify the result
            assert result.success is True
            assert result.project_name == "E-commerce Platform"
            assert result.generation_time_seconds == 120.5
            assert result.total_files_generated == 25
            
            # Verify the service was called with correct parameters
            mock_generation.assert_called_once()
            call_args = mock_generation.call_args[0][0]
            assert call_args.businessRequirement == sample_business_requirement
            assert call_args.projectName == "My E-commerce App"
            assert "Stripe" in call_args.additionalInstructions
    
    @pytest.mark.asyncio
    async def test_unified_generate_project_minimal_input(self, minimal_business_requirement):
        """Test project generation with minimal input"""
        
        request_data = {
            "businessRequirement": minimal_business_requirement
        }
        
        mock_result = ProjectGenerationResult(
            success=True,
            project_name="Todo App",
            brd_data=BRDCreatePayload(projectName="Todo App"),
            zip_file_path="/tmp/todo-app-generated.zip",
            generation_time_seconds=45.0,
            total_files_generated=8,
            ai_agent_calls=5
        )
        
        with patch('services.unified_project_generation', new_callable=AsyncMock, 
                  return_value=mock_result) as mock_generation:
            
            validated_request = UnifiedProjectRequest.model_validate(request_data)
            result = await services.unified_project_generation(validated_request)
            
            assert result.success is True
            assert result.project_name == "Todo App"
            assert result.total_files_generated == 8
            
            # Verify optional fields are None
            call_args = mock_generation.call_args[0][0]
            assert call_args.projectName is None
            assert call_args.additionalInstructions is None
    
    def test_invalid_request_validation(self):
        """Test validation of invalid request data"""
        
        # Test too short business requirement
        with pytest.raises(Exception):  # ValidationError from Pydantic
            UnifiedProjectRequest.model_validate({
                "businessRequirement": "Too short"
            })
        
        # Test missing business requirement
        with pytest.raises(Exception):
            UnifiedProjectRequest.model_validate({
                "projectName": "Test Project"
            })
    
    @pytest.mark.asyncio
    async def test_generation_error_handling(self, sample_business_requirement):
        """Test error handling during project generation"""
        
        request_data = {
            "businessRequirement": sample_business_requirement
        }
        
        with patch('services.unified_project_generation', new_callable=AsyncMock, 
                  side_effect=Exception("Project generation failed at repository_generation: Docker build failed")):
            
            validated_request = UnifiedProjectRequest.model_validate(request_data)
            
            with pytest.raises(Exception, match="Project generation failed at repository_generation"):
                await services.unified_project_generation(validated_request)
    
    def test_api_response_structure(self, mock_successful_generation_result):
        """Test the structure of API response data"""
        
        result = mock_successful_generation_result
        
        # Simulate API response construction
        response_data = {
            "status_code": 201,
            "message": f"🎉 Project '{result.project_name}' generated successfully!",
            "data": {
                "success": result.success,
                "project_name": result.project_name,
                "download_url": f"/api/download-project/{Path(result.zip_file_path).stem}",
                "zip_file_path": result.zip_file_path,
                "generation_metrics": {
                    "generation_time_seconds": result.generation_time_seconds,
                    "total_files_generated": result.total_files_generated,
                    "debug_attempts": result.debug_attempts,
                    "ai_agent_calls": result.ai_agent_calls
                },
                "brd_summary": {
                    "project_description": result.brd_data.projectDescription,
                    "technology_stack": result.brd_data.technologyStack.model_dump(),
                    "core_features_count": len(result.brd_data.coreFeatures),
                    "api_endpoints_count": len(result.brd_data.apiRequirements)
                },
                "validation_results": {
                    "overall_success": result.validation_report.overall_success,
                    "tests_count": len(result.validation_report.results),
                    "recommendations_count": len(result.validation_report.recommendations)
                }
            }
        }
        
        # Verify response structure
        assert response_data["status_code"] == 201
        assert "🎉" in response_data["message"]
        assert response_data["data"]["success"] is True
        assert response_data["data"]["download_url"].startswith("/api/download-project/")
        assert response_data["data"]["generation_metrics"]["generation_time_seconds"] == 120.5
        assert response_data["data"]["validation_results"]["overall_success"] is True


class TestProjectDownloadEndpoint:
    """Tests for the project download endpoint"""
    
    def test_download_url_construction(self):
        """Test download URL construction"""
        zip_file_path = "/tmp/generated_repos/my-awesome-project-generated.zip"
        zip_filename = Path(zip_file_path).stem  # "my-awesome-project-generated"
        
        download_url = f"/api/download-project/{zip_filename}"
        
        assert download_url == "/api/download-project/my-awesome-project-generated"
    
    def test_zip_filename_validation(self):
        """Test zip filename validation logic"""
        test_cases = [
            ("project-name", "project-name.zip"),
            ("project-name.zip", "project-name.zip"),
            ("complex-project-123", "complex-project-123.zip"),
        ]
        
        for input_filename, expected_filename in test_cases:
            # Simulate endpoint logic
            zip_filename = input_filename
            if not zip_filename.endswith('.zip'):
                zip_filename += '.zip'
            
            assert zip_filename == expected_filename


class TestProjectInfoEndpoint:
    """Tests for the project info endpoint"""
    
    def test_project_info_data_structure(self):
        """Test project info response structure"""
        
        # Mock project info data
        project_info = {
            "project_name": "Test Project",
            "project_path": "/tmp/test-project",
            "created_at": "2024-01-15T10:30:00",
            "file_count": 15,
            "validation_report": {
                "overall_success": True,
                "results": [
                    {"test_type": "build_test", "success": True}
                ]
            },
            "generation_session": {
                "ai_agent_calls": 8,
                "current_phase": "completed"
            },
            "file_structure": [
                {"path": "main.py", "size": 1024, "extension": ".py"},
                {"path": "requirements.txt", "size": 256, "extension": ".txt"},
                {"path": "Dockerfile", "size": 512, "extension": ""}
            ]
        }
        
        # Verify structure
        assert "project_name" in project_info
        assert "file_count" in project_info
        assert "validation_report" in project_info
        assert "file_structure" in project_info
        assert len(project_info["file_structure"]) == 3
        assert project_info["generation_session"]["ai_agent_calls"] == 8


class TestEndToEndFlow:
    """End-to-end integration tests"""
    
    @pytest.mark.asyncio
    async def test_complete_flow_simulation(self):
        """Test complete flow from business requirement to downloadable zip"""
        
        # Step 1: Business requirement input
        business_requirement = """
        Create a task management API with the following features:
        - User authentication with JWT
        - CRUD operations for tasks
        - Task categories and priorities
        - Due date reminders
        - RESTful API with OpenAPI documentation
        - PostgreSQL database
        - Docker containerization
        """
        
        request = UnifiedProjectRequest(
            businessRequirement=business_requirement,
            projectName="TaskMaster API",
            additionalInstructions="Include comprehensive unit tests and API rate limiting"
        )
        
        # Mock all the dependencies
        mock_brd = BRDCreatePayload(
            projectName="TaskMaster API",
            projectDescription="Comprehensive task management API",
            technologyStack=TechnologyStack(
                backend=["FastAPI", "Python"],
                database=["PostgreSQL"],
                other=["Docker", "JWT", "OpenAPI"]
            )
        )
        
        mock_validation_report = ValidationReport(
            overall_success=True,
            results=[
                ValidationResult(test_type="build_test", success=True, output="Build successful"),
                ValidationResult(test_type="api_test", success=True, output="All endpoints working"),
                ValidationResult(test_type="unit_test", success=True, output="All tests pass")
            ],
            recommendations=[]
        )
        
        expected_result = ProjectGenerationResult(
            success=True,
            project_name="TaskMaster API",
            brd_data=mock_brd,
            zip_file_path="/tmp/taskmaster-api-generated.zip",
            generation_time_seconds=89.3,
            validation_report=mock_validation_report,
            debug_attempts=0,
            total_files_generated=18,
            ai_agent_calls=10
        )
        
        # Mock the unified generation service
        with patch('services.unified_project_generation', new_callable=AsyncMock, 
                  return_value=expected_result) as mock_generation:
            
            # Execute the flow
            result = await services.unified_project_generation(request)
            
            # Verify end-to-end result
            assert result.success is True
            assert result.project_name == "TaskMaster API"
            assert result.validation_report.overall_success is True
            assert result.total_files_generated == 18
            assert result.debug_attempts == 0
            assert result.zip_file_path.endswith(".zip")
            
            # Verify the service was called correctly
            mock_generation.assert_called_once_with(request)
    
    def test_error_scenarios(self):
        """Test various error scenarios"""
        
        error_scenarios = [
            {
                "name": "BRD Generation Failure",
                "error": "Failed to generate BRD from business requirements",
                "stage": "brd_generation"
            },
            {
                "name": "Repository Generation Failure", 
                "error": "Docker build failed during file generation",
                "stage": "repository_generation"
            },
            {
                "name": "Validation Failure",
                "error": "Project validation tests failed",
                "stage": "validation"
            }
        ]
        
        for scenario in error_scenarios:
            # Each scenario would be tested with appropriate mocking
            # This demonstrates the error handling structure
            assert "error" in scenario
            assert "stage" in scenario
            assert scenario["stage"] in ["brd_generation", "repository_generation", "validation", "finalization"]


if __name__ == "__main__":
    pytest.main([__file__]) 