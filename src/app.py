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

async def validate_request_body(request: Request, model: type[M]) -> Union[M, ValidationError, ValueError]:
    data_for_validation = None
    preview_len = 200 # For logging previews of data

    try:
        print(f"DEBUG: validate_request_body START - Validating against model: {model.__name__}")

        # Step 1: Get raw request body
        raw_body = request.body # In Robyn, this is typically bytes for JSON requests
        print(f"DEBUG: validate_request_body - Step 1: `request.body` type: {type(raw_body)}")
        try:
            if isinstance(raw_body, bytes):
                print(f"DEBUG: validate_request_body - Step 1a: `request.body` (bytes preview {preview_len} chars): {raw_body[:preview_len]!r}")
            elif isinstance(raw_body, str):
                print(f"DEBUG: validate_request_body - Step 1a: `request.body` (string preview {preview_len} chars): {raw_body[:preview_len]!r}")
            elif raw_body is None:
                print("DEBUG: validate_request_body - Step 1a: `request.body` is None.")
            else: # dict, list, etc.
                print(f"DEBUG: validate_request_body - Step 1a: `request.body` (other type preview {preview_len} chars): {str(raw_body)[:preview_len]!r}")
        except Exception as log_ex:
            print(f"DEBUG: validate_request_body - Step 1b (LOGGING ERROR): Error creating preview of `request.body`: {log_ex}")

        if raw_body is None:
            print("DEBUG: validate_request_body - Step 2 ERROR: `request.body` is None. Cannot proceed.")
            return ValueError("Request body is missing.")

        # Step 2: Ensure body is a string for json.loads()
        body_str = None
        if isinstance(raw_body, bytes):
            try:
                body_str = raw_body.decode('utf-8')
                print(f"DEBUG: validate_request_body - Step 2: Decoded `request.body` from bytes to string. New type: {type(body_str)}")
            except UnicodeDecodeError as ude:
                print(f"DEBUG: validate_request_body - Step 2 ERROR: UTF-8 decoding failed: {ude}")
                return ValueError("Invalid request body encoding: Must be UTF-8.")
        elif isinstance(raw_body, str):
            body_str = raw_body
            print(f"DEBUG: validate_request_body - Step 2: `request.body` was already a string. Type: {type(body_str)}")
        elif isinstance(raw_body, dict): # If Robyn pre-parses it sometimes to dict
            print(f"DEBUG: validate_request_body - Step 2: `request.body` was already a dict. Using it directly for validation.")
            data_for_validation = raw_body
        else:
            print(f"DEBUG: validate_request_body - Step 2 ERROR: `request.body` is unexpected type: {type(raw_body)}. Expected bytes or str.")
            return ValueError(f"Unexpected request body type: {type(raw_body)}.")

        # Step 3: Parse string to dictionary if not already a dict
        if data_for_validation is None: # True if raw_body was bytes or str
            if body_str is None: # Should not happen if logic above is correct
                 print("DEBUG: validate_request_body - Step 3 LOGIC ERROR: body_str is None unexpectedly.")
                 return ValueError("Internal error processing request body string.")
            if not body_str.strip():
                 print("DEBUG: validate_request_body - Step 3 ERROR: Body string is empty or whitespace after decoding.")
                 return ValueError("Request body is empty.")

            print(f"DEBUG: validate_request_body - Step 3: Attempting `json.loads()` on body_str (preview {preview_len} chars): {body_str[:preview_len]!r}")
            try:
                data_for_validation = json.loads(body_str)
                print(f"DEBUG: validate_request_body - Step 3a: `json.loads()` successful. Type of `data_for_validation`: {type(data_for_validation)}")
            except json.JSONDecodeError as je:
                print(f"DEBUG: validate_request_body - Step 3b ERROR: `json.JSONDecodeError`: {je.msg} (line {je.lineno}, col {je.colno}, pos {je.pos})")
                print(f"DEBUG: validate_request_body - Failing JSON string (first {preview_len} chars for context): {body_str[:preview_len]!r}")
                return ValueError(f"Invalid JSON format in request body: {je.msg}")

        # Step 4: Validate the (hopefully) dictionary with Pydantic
        if data_for_validation is None:
            print("DEBUG: validate_request_body - Step 4 LOGIC ERROR: `data_for_validation` is None before Pydantic. This path should not be reached.")
            return ValueError("Failed to obtain processable dictionary from request body.")

        if not isinstance(data_for_validation, dict):
            # This check is crucial. If Pydantic still complains about string input, this check should have caught it.
            print(f"DEBUG: validate_request_body - Step 4 PRE-VALIDATION ERROR: `data_for_validation` is type {type(data_for_validation)}, NOT DICT.")
            print(f"DEBUG: validate_request_body - Content of `data_for_validation` (preview {preview_len} chars): {str(data_for_validation)[:preview_len]!r}")
            return ValueError(f"Request body parsed, but is not a JSON object (dictionary). Actual type: {type(data_for_validation)}.")

        print(f"DEBUG: validate_request_body - Step 4: PRE-PYDANTIC VALIDATION. `data_for_validation` type: {type(data_for_validation)}. Preview: {str(data_for_validation)[:preview_len]!r}")
        print(f"DEBUG: validate_request_body - Step 4a: Calling `{model.__name__}.model_validate(data_for_validation)`...")

        validated_model = model.model_validate(data_for_validation)
        print(f"DEBUG: validate_request_body - Step 5: Pydantic validation successful for {model.__name__}.")
        return validated_model

    except ValidationError as ve:
        # This catches errors from model.model_validate()
        print(f"DEBUG: validate_request_body - Pydantic ValidationError CAUGHT for model {model.__name__}:")
        # Pydantic V2: ve.errors() gives list of dicts. Each dict has 'input', 'type', 'loc', 'msg', 'ctx'.
        # The 'input' in each error detail shows what Pydantic received for that part of validation.
        # If the root model fails because input isn't a dict, the first error in ve.errors() should show that.
        errors_details = ve.errors(include_url=False)
        print(f"DEBUG: validate_request_body - Pydantic error details: {json.dumps(errors_details, indent=2)}")
        return ve
    except ValueError as val_err:
        # Catches ValueErrors explicitly raised by our logic (e.g., "Request body is empty.")
        print(f"DEBUG: validate_request_body - ValueError CAUGHT: {val_err}")
        return val_err
    except Exception as e:
        # Catches any other unexpected exceptions
        print(f"DEBUG: validate_request_body - Unexpected Exception CAUGHT for model {model.__name__}: {type(e).__name__}: {e}")
        import traceback
        print("DEBUG: validate_request_body - Traceback:")
        print(traceback.format_exc())
        return ValueError(f"An unexpected server error occurred while processing the request body: {type(e).__name__}.")



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
    validated_data = await validate_request_body(request, models.BRDTextCreatePayload)
    if isinstance(validated_data, (ValidationError, ValueError)):
        error_details = validated_data.errors() if isinstance(validated_data, ValidationError) else str(validated_data)
        return {
            "status_code": 400,
            "message": "Invalid data provided",
            "errors": error_details
        }


    try:
        generated_brd_payload = await services.generate_brd_from_text_with_ai(validated_data.businessInfo)

        new_brd = models.BRD(
            **generated_brd_payload.model_dump(),
            id=uuid4(),
            createdAt=datetime.utcnow(),
            updatedAt=datetime.utcnow()
        )

        print("--- Generated BRD Data from Text ---")
        print(json.dumps(new_brd.model_dump(mode='json'), indent=2))
        return {
            "status_code": 201,
            "message": "BRD created successfully from text",
            "data":  new_brd.model_dump(mode='json')
        }

    except Exception as e:
        print(f"Error in /api/brds/create-from-text: {e}")
        # Check if the error is from OpenAI or Pydantic validation of AI response
        return {
            "status_code": 500,
            "message": "Failed to generated BRD from text.",
            "error": str(e)
        }


@app.post("/api/brds/generate-repo")
async def generate_repo_endpoint(request: Request):
    validated_brd_data = await validate_request_body(request, models.BRDCreatePayload)
    if isinstance(validated_brd_data, (ValidationError, ValueError)):
        error_details = validated_brd_data.errors() if isinstance(validated_brd_data, ValidationError) else str(validated_brd_data)
        return {
            "status_code": 400,
            "message": "Invalid BRD data provided for repo generation.",
            "errors": error_details
        }


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
