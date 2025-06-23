import os
from robyn import Robyn, Request,  jsonify
from pydantic import ValidationError
from typing import TypeVar
from uuid import uuid4
from datetime import datetime
import json
from pathlib import Path

import models
import services

app = Robyn(__file__)

M = TypeVar('M', bound=models.BaseModel)



# @app.before_request()
# async def log_request(request: Request):
#     print(f"Received request: {request.method} {request.url.path}")


# @app.after_request()
# async def log_response(response: Response):
#     print(f"Sending response: Status {response.status_code}")



# --- API Endpoints ---
@app.get("/")
async def index(request: Request):
    return "DevBot Python Backend is running!"

@app.get("/health")
async def health(request: Request):
    return jsonify({
        "status": "ok",
        "message": "App is running!",
        "version": "0.1.0" # Example version
    })

@app.post("/api/brds")
async def create_brd_endpoint(request: Request):
    data = request.json()
    validated_data = models.BRDCreatePayload.model_validate(data);
    if isinstance(validated_data, (ValidationError, ValueError)):
        error_details = validated_data.errors() if isinstance(validated_data, ValidationError) else str(validated_data)
        return {
            "status_code": 400,
            "message": "Invalid data provided.",
            "errors": error_details
        }


    try:
        improved_brd_payload = await services.improve_brd_with_ai(validated_data)
        final_brd_data = improved_brd_payload

       # final_brd_data = validated_data
        new_brd = models.BRD(
            **final_brd_data.model_dump(),
            id=uuid4(),
            createdAt=datetime.utcnow(),
            updatedAt=datetime.utcnow()
        )

        print("--- Valid BRD Data Received & Processed ---")
        print(json.dumps(new_brd.model_dump(mode='json'), indent=2))
        return {
            "status_code":201,
            "message": "BRD Created successfully!",
            "data" : new_brd.model_dump(mode="json")
        }

    except Exception as e:
        print(f"Error in /api/brds: {e}")
        return {
            "status_code": 500,
            "message": "Failed to create BRD",
            "error": str(e)
        }

@app.post("/api/brds/create-from-text")
async def create_brd_from_text_endpoint(request: Request):
    data = request.json()
    validated_data = {}
    try:
        validated_data = models.BRDTextCreatePayload.model_validate(data)
        print(validated_data)
    except ValidationError as e:
        print(f"Validation error: {e}")
        return {
            "status_code": 400,
            "message": "Invalid request data",
            "error": str(e)
        }
    generated_brd_payload = await services.generate_brd_from_text_with_ai(validated_data.businessInfo)

    print("Received request data:", validated_data)

    return {
        "status_code": 201,
        "message": "BRD created successfully from text",
        "data":  generated_brd_payload.model_dump(mode='json')
    }

    # try:
    #

    #     new_brd = models.BRD(
    #         **generated_brd_payload.model_dump(),
    #         id=uuid4(),
    #         createdAt=datetime.utcnow(),
    #         updatedAt=datetime.utcnow()
    #     )

    #     print("--- Generated BRD Data from Text ---")
    #     print(json.dumps(new_brd.model_dump(mode='json'), indent=2))
    #     return {
    #         "status_code": 201,
    #         "message": "BRD created successfully from text",
    #         "data":  new_brd.model_dump(mode='json')
    #     }

    # except Exception as e:
    #     print(f"Error in /api/brds/create-from-text: {e}")
    #     return {
    #         "status_code": 500,
    #         "message": "Failed to generated BRD from text.",
    #         "error": str(e)
    #     }


@app.post("/api/brds/generate-repo")
async def generate_repo_endpoint(request: Request):
    data = request.json()
    validated_data = {}
    print(data)
    try:
        validated_data = models.BRDCreatePayload.model_validate(data)
    except ValueError as ve:
        print(f"Validation error in /api/brds/generate-repo: {ve}")

        return {
            "message" : f"{ve}",
            "status_code": 400
        }
    except Exception as e:
        print(f"Error in /api/brds/generate-repo: {e}")
        return {
            "status_code": 500,
            "message": "Failed to generate repository structure.",
            "error": str(e)
        }
    validated_brd_data = validated_data
    try:
        print(f"\n=== Starting Enhanced Repository Generation ===")
        print(f"Project: {validated_brd_data.projectName}")
        print(f"Technology Stack: {validated_brd_data.technologyStack.model_dump()}")
        
        start_time = datetime.utcnow()
        project_path = await services.generate_repository_files(validated_brd_data)
        end_time = datetime.utcnow()
        
        generation_time = (end_time - start_time).total_seconds()
        
        # Check if validation report exists
        validation_report_path = Path(project_path) / "validation_report.json"
        validation_summary = None
        
        if validation_report_path.exists():
            try:
                with open(validation_report_path, 'r') as f:
                    validation_data = json.loads(f.read())
                    validation_summary = {
                        "overall_success": validation_data.get("overall_success", False),
                        "total_tests": len(validation_data.get("results", [])),
                        "passed_tests": len([r for r in validation_data.get("results", []) if r.get("success", False)]),
                        "recommendations_count": len(validation_data.get("recommendations", []))
                    }
            except Exception as e:
                print(f"Could not read validation report: {e}")
        
        response_data = {
            "status_code": 201,
            "message": f"Enhanced repository generation completed for '{validated_brd_data.projectName}'",
            "data": {
                "project_name": validated_brd_data.projectName,
                "project_path": project_path,
                "generation_time_seconds": generation_time,
                "validation_summary": validation_summary,
                "technology_stack": validated_brd_data.technologyStack.model_dump(),
                "features_count": len(validated_brd_data.coreFeatures),
                "api_endpoints_count": len(validated_brd_data.apiRequirements)
            }
        }
        
        print(f"=== Repository Generation Completed Successfully ===")
        print(f"Total time: {generation_time:.2f} seconds")
        if validation_summary:
            print(f"Validation: {validation_summary['passed_tests']}/{validation_summary['total_tests']} tests passed")
        
        return response_data
        
    except ValueError as ve: # Specific errors like slug issues, circular deps
         print(f"Validation error in /api/brds/generate-repo: {ve}")

         return {
             "message" : f"{ve}",
             "status_code": 400
         }
    except Exception as e:
        print(f"Error in /api/brds/generate-repo: {e}")
        return {
            "status_code": 500,
            "message": "Failed to generate repository structure with enhanced debugging and validation.",
            "error": str(e)
        }

@app.get("/api/brds/generation-status/{project_name}")
async def get_generation_status(request: Request):
    """
    Get detailed status of repository generation including AI agent calls and debugging history.
    """
    project_name = request.path_params.get("project_name")
    if not project_name:
        return {
            "status_code": 400,
            "message": "Project name is required"
        }
    
    try:
        # Construct the project path
        project_name_slug = project_name.lower().replace(" ", "-").replace(r"[^a-z0-9-]", "")
        base_repo_path = os.path.join(os.path.dirname(__file__), "../../generated_repos")
        project_path = Path(base_repo_path) / project_name_slug
        
        if not project_path.exists():
            return {
                "status_code": 404,
                "message": f"Project '{project_name}' not found"
            }
        
        # Read validation report if it exists
        validation_report_path = project_path / "validation_report.json"
        validation_report = None
        if validation_report_path.exists():
            with open(validation_report_path, 'r') as f:
                validation_report = json.loads(f.read())
        
        # Get project structure
        project_files = []
        for root, _, files in os.walk(project_path):
            for file in files:
                if file != "validation_report.json":
                    file_path = Path(root) / file
                    relative_path = file_path.relative_to(project_path)
                    project_files.append(str(relative_path))
        
        return {
            "status_code": 200,
            "message": "Project status retrieved successfully",
            "data": {
                "project_name": project_name,
                "project_path": str(project_path),
                "project_files": project_files,
                "validation_report": validation_report,
                "file_count": len(project_files)
            }
        }
    
    except Exception as e:
        print(f"Error retrieving project status: {e}")
        return {
            "status_code": 500,
            "message": "Failed to retrieve project status",
            "error": str(e)
        }


@app.post("/api/brds/debug-project")
async def debug_project_endpoint(request: Request):
    """
    Manually trigger debugging for an existing project.
    """
    data = request.json()
    project_name = data.get("projectName")
    force_rebuild = data.get("forceRebuild", False)
    
    if not project_name:
        return {
            "status_code": 400,
            "message": "Project name is required"
        }
    
    try:
        # Construct the project path
        project_name_slug = project_name.lower().replace(" ", "-").replace(r"[^a-z0-9-]", "")
        base_repo_path = os.path.join(os.path.dirname(__file__), "../../generated_repos")
        project_path = Path(base_repo_path) / project_name_slug
        
        if not project_path.exists():
            return {
                "status_code": 404,
                "message": f"Project '{project_name}' not found"
            }
        
        print(f"\n=== Manual Debug Session for {project_name} ===")
        
        # Try to build the project
        success, logs = await services.run_docker_based_tests(project_path, project_name_slug)
        
        if success and not force_rebuild:
            return {
                "status_code": 200,
                "message": "Project builds successfully, no debugging needed",
                "data": {
                    "build_success": True,
                    "build_logs": logs
                }
            }
        
        # Create a minimal BRD for context (this is a simplified version)
        minimal_brd = models.BRDCreatePayload(
            projectName=project_name,
            projectDescription="Manual debugging session",
            technologyStack=models.TechnologyStack(),
            coreFeatures=[],
            dataModels=[],
            apiRequirements=[]
        )
        
        # Initialize debugging session
        session = models.MultiAgentSession(
            project_context=f"Manual debugging for: {project_name}",
            current_phase="manual_debugging"
        )
        
        debug_session = models.DebugSession(
            project_path=str(project_path),
            max_attempts=3
        )
        
        # Run debugging
        modified_files = await services.enhanced_debug_code_with_ai(
            project_path, minimal_brd, logs, debug_session, session
        )
        
        # Test again after debugging
        final_success, final_logs = await services.run_docker_based_tests(project_path, project_name_slug)
        
        return {
            "status_code": 200,
            "message": "Manual debugging completed",
            "data": {
                "initial_build_success": success,
                "final_build_success": final_success,
                "modified_files": modified_files,
                "debug_attempts": len(debug_session.attempts),
                "ai_agent_calls": len(session.agent_calls),
                "final_logs": final_logs
            }
        }
    
    except Exception as e:
        print(f"Error in manual debugging: {e}")
        return {
            "status_code": 500,
            "message": "Manual debugging failed",
            "error": str(e)
        }

if __name__ == "__main__":
    base_repo_path = os.path.join(os.path.dirname(__file__), "../../generated_repos")
    os.makedirs(base_repo_path, exist_ok=True)
    print(f"Base path for generated repositories: {os.path.abspath(base_repo_path)}")

    app.start(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
