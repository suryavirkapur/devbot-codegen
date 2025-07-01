import os
from robyn import Robyn, Request, serve_file
from pydantic import ValidationError
from pathlib import Path
import json

import models
from worker import celery_app, generate_project_task

app = Robyn(__file__)

@app.post("/api/generate-project")
def unified_generate_project_endpoint(request: Request):
    """
    🚀 Kicks off an async project generation task 🚀
    
    Takes a business requirement as text/markdown and starts the generation process.
    Returns a task ID to check the status of the generation.
    """
    try:
        data = request.json()
        
        # Basic validation before queueing
        try:
            models.UnifiedProjectRequest.model_validate(data)
        except ValidationError as e:
            return {
                "status_code": 400,
                "message": "Invalid request data",
                "errors": e.errors()
            }
        
        # Dispatch the task to the Celery worker
        task = generate_project_task.delay(data)
        
        print(f"Dispatched project generation task with ID: {task.id}")
        
        return {
            "status_code": 202,
            "message": "Project generation has been started.",
            "task_id": task.id,
            "status_url": f"/api/generation-status/{task.id}"
        }
        
    except Exception as e:
        print(f"❌ Error dispatching task: {e}")
        return {
            "status_code": 500,
            "message": "Failed to start project generation task",
            "error": str(e)
        }

@app.get("/api/generation-status/<task_id>")
def get_generation_status(request: Request):
    """
    Checks the status of a project generation task.
    """
    task_id = request.path_params.get("task_id")
    task_result = celery_app.AsyncResult(task_id)
    
    response = {
        "task_id": task_id,
        "status": task_result.status,
        "result": None
    }
    
    if task_result.successful():
        result_json = task_result.get()
        result_data = json.loads(result_json)
        response["result"] = result_data
        response["message"] = f"🎉 Project '{result_data.get('project_name')}' generated successfully!"
        response["download_url"] = f"/api/download-project/{Path(result_data.get('zip_file_path')).stem}"

    elif task_result.failed():
        # The result of a failed task is the exception that was raised.
        # We should be careful about what we expose to the client.
        error_info = {
            "error": str(task_result.result),
            "traceback": task_result.traceback
        }
        response["result"] = error_info
        response["message"] = "Project generation failed."

    return response


@app.get("/api/download-project/{zip_filename}")
async def download_project_zip(request: Request):
    """
    Download the generated project zip file.
    """
    zip_filename = request.path_params.get("zip_filename")
    if not zip_filename:
        return {
            "status_code": 400,
            "message": "Zip filename is required"
        }
    
    # Ensure .zip extension
    if not zip_filename.endswith('.zip'):
        zip_filename += '.zip'
    
    try:
        # Look for the zip file in the generated repos directory
        base_repo_path = Path(os.path.dirname(__file__)).parent.parent / "generated_repos"
        zip_path = base_repo_path / zip_filename
        
        if not zip_path.exists():
            return {
                "status_code": 404,
                "message": f"Zip file '{zip_filename}' not found"
            }
        
        print(f"📦 Serving zip file: {zip_path}")
        
        # Serve the file using Robyn's serve_file function
        return serve_file(str(zip_path))
        
    except Exception as e:
        print(f"Error serving zip file: {e}")
        return {
            "status_code": 500,
            "message": "Failed to serve zip file",
            "error": str(e)
        }

if __name__ == "__main__":
    base_repo_path = os.path.join(os.path.dirname(__file__), "../../generated_repos")
    os.makedirs(base_repo_path, exist_ok=True)
    print(f"Base path for generated repositories: {os.path.abspath(base_repo_path)}")

    app.start(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
