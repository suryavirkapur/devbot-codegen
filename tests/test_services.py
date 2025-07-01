import pytest
import asyncio
import tempfile
import shutil
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import json

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import (
    BRDCreatePayload, TechnologyStack, CoreFeature, UnifiedProjectRequest,
    ProjectGenerationResult, FileDependencyInfo, DependencyOrder,
    ValidationTest, ValidationPlan, ValidationResult, ValidationReport,
    MultiAgentSession
)
import services


class TestFileSummarization:
    @pytest.mark.asyncio
    async def test_summarize_file_content_python(self):
        """Test summarization of Python files"""
        python_content = '''
import os
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
async def health_check():
    return {"status": "ok"}

class UserService:
    def __init__(self):
        self.users = []
    
    def create_user(self, name: str):
        user = {"id": len(self.users), "name": name}
        self.users.append(user)
        return user
'''
        
        with patch('services.client.chat.completions.create') as mock_create:
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = """
            This is a FastAPI application with:
            1. Health check endpoint at /health
            2. UserService class for managing users
            3. Basic CRUD operations for user management
            4. Uses FastAPI framework for REST API
            """
            mock_create.return_value = mock_response
            
            result = await services.summarize_file_content("src/main.py", python_content)
            
            assert "FILE SUMMARY: src/main.py" in result
            assert "FastAPI" in result or "health check" in result.lower()
            mock_create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_summarize_file_content_json(self):
        """Test summarization of JSON configuration files"""
        json_content = '''
{
    "name": "my-app",
    "version": "1.0.0",
    "scripts": {
        "start": "node index.js",
        "test": "jest"
    },
    "dependencies": {
        "express": "^4.18.0",
        "mongoose": "^6.0.0"
    }
}
'''
        
        with patch('services.client.chat.completions.create') as mock_create:
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = """
            Package.json file for Node.js application:
            1. App name: my-app, version 1.0.0
            2. Dependencies: Express for web server, Mongoose for MongoDB
            3. Scripts: start with node, testing with jest
            """
            mock_create.return_value = mock_response
            
            result = await services.summarize_file_content("package.json", json_content)
            
            assert "FILE SUMMARY: package.json" in result
            mock_create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_summarize_file_content_error_handling(self):
        """Test error handling in file summarization"""
        
        with patch('services.client.chat.completions.create', side_effect=Exception("API Error")):
            result = await services.summarize_file_content("test.py", "print('hello')")
            
            assert "FILE: test.py" in result
            assert "Content length:" in result
            assert ".py" in result


class TestTopologicalSorting:
    def test_topological_sort_simple(self):
        """Test simple topological sorting"""
        files = [
            FileDependencyInfo(
                path="config.py",
                description="Configuration file",
                dependsOn=[]
            ),
            FileDependencyInfo(
                path="main.py",
                description="Main application",
                dependsOn=["config.py"]
            )
        ]
        
        sorted_files = services.topological_sort_files(files)
        
        assert len(sorted_files) == 2
        # config.py should come before main.py
        paths = [f.path for f in sorted_files]
        assert paths.index("config.py") < paths.index("main.py")
    
    def test_topological_sort_complex(self):
        """Test complex dependency graph"""
        files = [
            FileDependencyInfo(path="app.py", description="App", dependsOn=["models.py", "config.py"]),
            FileDependencyInfo(path="models.py", description="Models", dependsOn=["config.py"]),
            FileDependencyInfo(path="config.py", description="Config", dependsOn=[]),
            FileDependencyInfo(path="utils.py", description="Utils", dependsOn=[])
        ]
        
        sorted_files = services.topological_sort_files(files)
        paths = [f.path for f in sorted_files]
        
        # config.py should come before models.py and app.py
        assert paths.index("config.py") < paths.index("models.py")
        assert paths.index("models.py") < paths.index("app.py")
        # utils.py can be anywhere since it has no dependencies
    
    def test_circular_dependency_detection(self):
        """Test detection of circular dependencies"""
        files = [
            FileDependencyInfo(path="a.py", description="A", dependsOn=["b.py"]),
            FileDependencyInfo(path="b.py", description="B", dependsOn=["a.py"])
        ]
        
        with pytest.raises(ValueError, match="Circular dependency"):
            services.topological_sort_files(files)


class TestZipFileCreation:
    def test_create_project_zip(self):
        """Test project zip file creation"""
        # Create a temporary directory with some files
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir) / "test_project"
            project_path.mkdir()
            
            # Create some test files
            (project_path / "main.py").write_text("print('hello')")
            (project_path / "README.md").write_text("# Test Project")
            
            subdir = project_path / "src"
            subdir.mkdir()
            (subdir / "app.py").write_text("from fastapi import FastAPI")
            
            # Create zip
            zip_path = services.create_project_zip(project_path, "test-project")
            
            # Verify zip was created
            assert Path(zip_path).exists()
            assert zip_path.endswith("test-project.zip")
            
            # Verify zip contents
            with zipfile.ZipFile(zip_path, 'r') as zipf:
                files_in_zip = zipf.namelist()
                assert "main.py" in files_in_zip
                assert "README.md" in files_in_zip
                assert "src/app.py" in files_in_zip


class TestUnifiedProjectGeneration:
    @pytest.mark.asyncio
    async def test_unified_project_generation_success(self):
        """Test successful unified project generation"""
        request = UnifiedProjectRequest(
            businessRequirement="Create a simple todo app with FastAPI and SQLite",
            projectName="Todo App",
            additionalInstructions="Include Docker support"
        )
        
        # Mock all the service calls
        mock_brd = BRDCreatePayload(
            projectName="Todo App",
            projectDescription="A simple todo application",
            technologyStack=TechnologyStack(
                backend=["FastAPI", "Python"],
                database=["SQLite"]
            ),
            coreFeatures=[
                CoreFeature(
                    name="Task Management",
                    description="Create, read, update, delete tasks",
                    priority="High"
                )
            ]
        )
        
        with patch('services.generate_brd_from_text_with_ai', new_callable=AsyncMock) as mock_brd_gen, \
             patch('services.improve_brd_with_ai', new_callable=AsyncMock) as mock_brd_improve, \
             patch('services.generate_repository_files', new_callable=AsyncMock) as mock_repo_gen, \
             patch('services.create_project_zip') as mock_zip, \
             patch('pathlib.Path.exists', return_value=False), \
             patch('pathlib.Path.rglob') as mock_rglob:
            
            mock_brd_gen.return_value = mock_brd
            mock_brd_improve.return_value = mock_brd
            mock_repo_gen.return_value = "/tmp/todo-app"
            mock_zip.return_value = "/tmp/todo-app.zip"
            mock_rglob.return_value = [Mock() for _ in range(10)]  # 10 files
            
            result = await services.unified_project_generation(request)
            
            assert isinstance(result, ProjectGenerationResult)
            assert result.success is True
            assert result.project_name == "Todo App"
            assert result.total_files_generated == 10
            assert result.zip_file_path == "/tmp/todo-app.zip"
            
            # Verify all steps were called
            mock_brd_gen.assert_called_once_with(request.businessRequirement)
            mock_brd_improve.assert_called_once()
            mock_repo_gen.assert_called_once()
            mock_zip.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_unified_project_generation_with_validation_report(self):
        """Test unified generation with validation report loading"""
        request = UnifiedProjectRequest(
            businessRequirement="Create a REST API with authentication"
        )
        
        mock_brd = BRDCreatePayload(projectName="API Project")
        
        # Mock validation report
        validation_data = {
            "overall_success": True,
            "results": [
                {"test_type": "build_test", "success": True, "output": "Build successful"}
            ],
            "recommendations": []
        }
        
        with patch('services.generate_brd_from_text_with_ai', new_callable=AsyncMock, return_value=mock_brd), \
             patch('services.improve_brd_with_ai', new_callable=AsyncMock, return_value=mock_brd), \
             patch('services.generate_repository_files', new_callable=AsyncMock, return_value="/tmp/api-project"), \
             patch('services.create_project_zip', return_value="/tmp/api-project.zip"), \
             patch('pathlib.Path.exists', side_effect=lambda p: "validation_report.json" in str(p)), \
             patch('pathlib.Path.rglob', return_value=[Mock() for _ in range(5)]), \
             patch('builtins.open', mock_open_with_json(validation_data)):
            
            result = await services.unified_project_generation(request)
            
            assert result.success is True
            assert result.validation_report is not None
            assert result.validation_report.overall_success is True
    
    @pytest.mark.asyncio
    async def test_unified_project_generation_error_handling(self):
        """Test error handling in unified project generation"""
        request = UnifiedProjectRequest(
            businessRequirement="Create an app that will definitely fail"
        )
        
        with patch('services.generate_brd_from_text_with_ai', new_callable=AsyncMock, 
                  side_effect=Exception("BRD generation failed")):
            
            with pytest.raises(Exception, match="Project generation failed at brd_generation"):
                await services.unified_project_generation(request)


class TestBRDGeneration:
    @pytest.mark.asyncio
    async def test_generate_brd_from_text_with_ai(self):
        """Test BRD generation from text"""
        business_info = "I need a blog platform where users can write posts, comment, and follow each other"
        
        mock_brd = BRDCreatePayload(
            projectName="Blog Platform",
            projectDescription="Social blogging platform",
            technologyStack=TechnologyStack(
                frontend=["React"],
                backend=["FastAPI"],
                database=["PostgreSQL"]
            )
        )
        
        with patch('services.generate_structured_openai_response', new_callable=AsyncMock, 
                  return_value=mock_brd) as mock_openai:
            
            result = await services.generate_brd_from_text_with_ai(business_info)
            
            assert isinstance(result, BRDCreatePayload)
            assert result.projectName == "Blog Platform"
            mock_openai.assert_called_once()
            
            # Check that the prompt contains the business info
            call_args = mock_openai.call_args
            assert business_info in call_args[0][0]  # First positional argument (prompt)
    
    @pytest.mark.asyncio
    async def test_improve_brd_with_ai(self):
        """Test BRD improvement with AI"""
        original_brd = BRDCreatePayload(
            projectName="Basic App",
            projectDescription="Simple app"
        )
        
        improved_brd = BRDCreatePayload(
            projectName="Enhanced App",
            projectDescription="A comprehensive application with advanced features",
            technologyStack=TechnologyStack(backend=["FastAPI"])
        )
        
        with patch('services.generate_structured_openai_response', new_callable=AsyncMock, 
                  return_value=improved_brd) as mock_openai:
            
            result = await services.improve_brd_with_ai(original_brd)
            
            assert result.projectName == "Enhanced App"
            assert len(result.projectDescription) > len(original_brd.projectDescription)
            mock_openai.assert_called_once()


class TestValidationPlanAndExecution:
    @pytest.mark.asyncio
    async def test_create_validation_plan(self):
        """Test validation plan creation"""
        brd_data = BRDCreatePayload(
            projectName="Test API",
            technologyStack=TechnologyStack(backend=["FastAPI"])
        )
        
        session = MultiAgentSession(project_context="Test")
        
        mock_plan = ValidationPlan(
            tests=[
                ValidationTest(
                    test_type="build_test",
                    test_command="docker build .",
                    timeout=120
                ),
                ValidationTest(
                    test_type="api_test",
                    test_command="curl http://localhost:8080/health",
                    expected_output="ok"
                )
            ],
            setup_commands=["pip install -r requirements.txt"],
            cleanup_commands=["docker system prune -f"]
        )
        
        with tempfile.TemporaryDirectory() as temp_dir, \
             patch('services.call_ai_agent', new_callable=AsyncMock, 
                   return_value=(mock_plan, Mock())) as mock_agent:
            
            project_path = Path(temp_dir)
            (project_path / "main.py").write_text("from fastapi import FastAPI")
            
            result = await services.create_validation_plan(project_path, brd_data, session)
            
            assert isinstance(result, ValidationPlan)
            assert len(result.tests) == 2
            assert any(test.test_type == "build_test" for test in result.tests)
            mock_agent.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_run_validation_tests(self):
        """Test validation test execution"""
        validation_plan = ValidationPlan(
            tests=[
                ValidationTest(
                    test_type="echo_test",
                    test_command="echo 'success'",
                    expected_output="success",
                    timeout=10
                )
            ],
            setup_commands=[],
            cleanup_commands=[]
        )
        
        session = MultiAgentSession(project_context="Test")
        
        with tempfile.TemporaryDirectory() as temp_dir, \
             patch('services.call_ai_agent', new_callable=AsyncMock, 
                   return_value=([], Mock())) as mock_agent:
            
            project_path = Path(temp_dir)
            
            result = await services.run_validation_tests(project_path, validation_plan, session)
            
            assert isinstance(result, ValidationReport)
            assert len(result.results) == 1
            assert result.results[0].test_type == "echo_test"
            assert result.results[0].success is True
            assert result.overall_success is True


def mock_open_with_json(json_data):
    """Helper function to mock file opening with JSON data"""
    import io
    from unittest.mock import mock_open
    
    return mock_open(read_data=json.dumps(json_data))


if __name__ == "__main__":
    pytest.main([__file__]) 