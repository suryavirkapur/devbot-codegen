import json
import os
import shutil

from pathlib import Path
from typing import TypeVar, Type, List, Dict
from openai import OpenAI, AzureOpenAI
import config

import models

if os.getenv("AZURE_OPENAI_ENDPOINT"): # Example condition to use Azure
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT") or ""
    )

else:
    client = OpenAI(api_key=config.OPENAI_API_KEY, base_url="https://openai-proxy.svk77.com")


T = TypeVar('T', bound=models.BaseModel)

async def generate_structured_openai_response(prompt: str, response_model: Type[T], model_name: str = config.DEFAULT_MODEL) -> T:
    """
    Generates a structured response from OpenAI and validates it against the Pydantic model.
    """
    try:

        completion = client.responses.parse(
            model=model_name,
            input=[
                {"role": "system", "content": f"You are a helpful assistant. Please respond in JSON format matching the following Pydantic schema: {response_model.model_json_schema()}"},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"} # Use JSON mode
        )

        content = completion.choices[0].message.content

        if not content:
            raise ValueError("OpenAI returned empty content.")

        try:
            json_data = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"Failed to parse AI response as JSON: {content}")
            raise ValueError(f"AI response was not valid JSON. Error: {e}") from e

        return response_model.model_validate(json_data)

    except Exception as e:
        print(f"Error in generate_structured_openai_response: {e}")
        # Consider re-raising a custom error or handling it
        raise

async def generate_code_from_prompt(prompt: str, model_name: str = config.DEFAULT_MODEL) -> str:
    """
    Generates code or text content from a prompt using OpenAI.
    """
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a code generation assistant. Output only the raw code/text as requested, without any explanations or markdown formatting unless it's part of the requested file content itself."},
                {"role": "user", "content": prompt}
            ]
        )
        content = completion.choices[0].message.content
        if not content:
            return "" # Return empty string if no content
        # Remove potential markdown code block fences if AI adds them
        if content.startswith("```") and content.endswith("```"):
            content = content.splitlines(True)[1:-1] # Keep newlines
            content = "".join(content)
        return content.strip()
    except Exception as e:
        print(f"Error in generate_code_from_prompt: {e}")
        raise


async def improve_brd_with_ai(brd_payload: models.BRDCreatePayload) -> models.BRDCreatePayload:
    prompt = f"""
    Please review and improve the following Business Requirements Document (BRD) draft.
    Enhance clarity, completeness, and consistency. Ensure all fields are well-defined.
    If some features or data models are too vague, try to make them more specific or add example details.

    Current BRD Draft:
    {brd_payload.model_dump_json(indent=2)}

    Return the improved BRD as a JSON object matching the BRDCreatePayload schema.
    """
    improved_brd_data = await generate_structured_openai_response(prompt, models.BRDCreatePayload)
    return improved_brd_data


async def generate_brd_from_text_with_ai(business_info: str) -> models.BRDCreatePayload:
    prompt = f"""
    Based on the following business information, generate a structured Business Requirements Document (BRD).
    Fill out all the fields of the BRD schema based on the provided text.

    Business Information:
    {business_info}

    Generate a complete BRD and return it as a JSON object matching the BRDCreatePayload schema.
    """
    generated_brd_data = await generate_structured_openai_response(prompt, models.BRDCreatePayload)
    return generated_brd_data


# --- Repository Generation Logic ---

def ensure_directory_exists(dir_path: Path):
    dir_path.mkdir(parents=True, exist_ok=True)
    print(f"Directory ensured: {dir_path}")

def topological_sort_files(files: List[models.FileDependencyInfo]) -> List[models.FileDependencyInfo]:
    sorted_list: List[models.FileDependencyInfo] = []
    visited = set()  # Stores paths of files whose processing is complete
    visiting = set() # Stores paths of files currently in recursion stack for cycle detection

    file_map: Dict[str, models.FileDependencyInfo] = {f.path: f for f in files}

    def visit(file_info: models.FileDependencyInfo):
        if file_info.path in visited:
            return
        if file_info.path in visiting:
            raise ValueError(f"Circular dependency detected involving: {file_info.path}")

        visiting.add(file_info.path)

        for dep_path in file_info.dependsOn:
            dep_file = file_map.get(dep_path)
            if dep_file:
                visit(dep_file)
            else:
                print(f"Warning: Dependency '{dep_path}' for file '{file_info.path}' not found. Skipping.")

        visiting.remove(file_info.path)
        visited.add(file_info.path)
        sorted_list.append(file_info)

    for file_info in files:
        if file_info.path not in visited:
            visit(file_info)

    return sorted_list


async def generate_repository_files(brd_data: models.BRDCreatePayload) -> str:
    project_name_slug = brd_data.projectName.lower().replace(" ", "-").replace(r"[^a-z0-9-]", "")
    if not project_name_slug:
        raise ValueError("Project name is invalid or results in an empty slug.")

    project_path = Path(config.BASE_GENERATED_REPOS_PATH) / project_name_slug

    ensure_directory_exists(Path(config.BASE_GENERATED_REPOS_PATH))

    if project_path.exists():
        print(f"Project directory {project_path} already exists. Removing existing directory.")
        shutil.rmtree(project_path)

    project_path.mkdir(parents=True)
    print(f"Created project directory: {project_path}")

    # Step 1: Generate project file structure with dependencies
    structure_prompt = f"""
    Based on the following Business Requirements Document (BRD), define a complete project file structure.
    For each file, specify its full path (relative to project root), a detailed description of its content/purpose,
    and an array of other file paths it depends on ('dependsOn').
    Ensure the file order considers dependencies (e.g., utility files before files that use them).

    BRD:
    {brd_data.model_dump_json(indent=2)}

    Return the file structure as a JSON object matching the DependencyOrder schema.
    """
    dependency_structure = await generate_structured_openai_response(structure_prompt, models.DependencyOrder)
    print("--- Generated Project Structure with Dependencies ---")
    print(dependency_structure.model_dump_json(indent=2))

    # Step 2: Topologically sort files
    try:
        sorted_files = topological_sort_files(dependency_structure.files)
        print("--- Topologically Sorted File Generation Order ---")
        print(" -> ".join([f.path for f in sorted_files]))
    except ValueError as e:
        print(f"Error in sorting files: {e}")
        raise ValueError(f"Could not generate repository due to dependency issue: {e}")


    # Step 3: Generate each file
    generated_file_contents: Dict[str, str] = {}
    brd_string = brd_data.model_dump_json(indent=2)

    for file_info in sorted_files:
        context_parts = [
            f"### FILE: {path}\n```\n{content[:2500]}...\n```" # Truncate for context
            for path, content in generated_file_contents.items() if path in file_info.dependsOn # Only provide direct dependencies as context for now
        ]
        context_string = "\n\n".join(context_parts) if context_parts else "No relevant files have been created yet or no direct dependencies listed for this file."

        file_gen_prompt = f"""
        Generate the complete and raw source code/content for the file: {file_info.path}

        File Description:
        {file_info.description}

        Overall Project BRD (for context):
        {brd_string}

        Relevant Previously Generated Dependent Files (for context and correct imports/references):
        {context_string}

        Important: Output ONLY the raw content for the file '{file_info.path}'.
        Do not include any explanations, comments about the code, or markdown code block fences unless they are part of the file's actual content (e.g. a Markdown file).
        """

        print(f"Generating content for {file_info.path}...")
        file_content = await generate_code_from_prompt(file_gen_prompt)

        full_file_path = project_path / file_info.path
        full_file_path.parent.mkdir(parents=True, exist_ok=True) # Ensure parent directory exists

        with open(full_file_path, "w", encoding="utf-8") as f:
            f.write(file_content)

        print(f"Successfully created file: {full_file_path}")
        generated_file_contents[file_info.path] = file_content

    return str(project_path)
