
import os
from robyn import Robyn, Request,  jsonify
from pydantic import ValidationError
from typing import TypeVar, Union
from uuid import uuid4
from datetime import datetime
import json

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
    validated_data = await validate_request_body(request, models.BRDCreatePayload)
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
        project_path = await services.generate_repository_files(validated_brd_data)
        return {
            "status_code": 201,
            "message": f"Project '{validated_brd_data}' generated!",
            "path": project_path
        }
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
            "message": "Failed to generate repository structure.",
            "error": str(e)
        }


if __name__ == "__main__":
    base_repo_path = os.path.join(os.path.dirname(__file__), "../../generated_repos")
    os.makedirs(base_repo_path, exist_ok=True)
    print(f"Base path for generated repositories: {os.path.abspath(base_repo_path)}")

    app.start(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
