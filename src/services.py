import os
import shutil
import subprocess
import asyncio
import json
import time
from datetime import datetime

from pathlib import Path
from typing import TypeVar, Type, List, Dict, Optional, Tuple
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
    client = OpenAI(api_key=config.OPENAI_API_KEY, base_url="https://openai-proxy.svk77.com/v1")


T = TypeVar('T', bound=models.BaseModel)

async def generate_structured_openai_response(prompt: str, response_model: Type[T], model_name: str = config.DEFAULT_MODEL) -> T:
    """
    Generates a structured response from OpenAI and validates it against the Pydantic model.
    """
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": f"You are a helpful assistant. Please respond in JSON format matching the following Pydantic schema: {response_model.model_json_schema()}"},
                {"role": "user", "content": prompt}
            ]
        )

        content = completion.choices[0].message.content

        if not content:
            raise ValueError("OpenAI returned empty content.")

        # Parse the JSON response
        import json
        parsed_content = json.loads(content)
        return response_model.model_validate(parsed_content)

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

async def summarize_file_content(file_path: str, content: str, model_name: str = config.DEFAULT_MODEL) -> str:
    """
    Creates an intelligent summary of file content for context inclusion.
    """
    # Determine file type for specialized summarization
    file_extension = Path(file_path).suffix.lower()
    
    if file_extension in ['.py', '.js', '.ts', '.go', '.rs', '.java', '.cpp', '.c']:
        system_prompt = """You are a code analysis assistant. Summarize the provided code file by extracting:
        1. Main purpose and functionality
        2. Key classes, functions, and their purposes
        3. Important imports and dependencies
        4. API endpoints or main entry points
        5. Key data structures or models
        Keep the summary concise but comprehensive (200-400 words max)."""
    elif file_extension in ['.json', '.yaml', '.yml', '.toml']:
        system_prompt = """You are a configuration analysis assistant. Summarize the configuration file by extracting:
        1. Main configuration purpose
        2. Key settings and their values
        3. Dependencies or packages listed
        4. Important environment variables or secrets
        Keep the summary concise (150-250 words max)."""
    elif file_extension in ['.md', '.txt', '.rst']:
        system_prompt = """You are a documentation analysis assistant. Summarize the document by extracting:
        1. Main topic and purpose
        2. Key sections and their content
        3. Important instructions or commands
        4. Setup or usage information
        Keep the summary concise (200-300 words max)."""
    elif file_extension in ['.dockerfile', 'dockerfile']:
        system_prompt = """You are a Docker analysis assistant. Summarize the Dockerfile by extracting:
        1. Base image and runtime environment
        2. Key installation steps
        3. Exposed ports and volumes
        4. Entry point and command
        Keep the summary concise (150-250 words max)."""
    else:
        system_prompt = """You are a file analysis assistant. Summarize the file content by extracting:
        1. Main purpose and type of content
        2. Key sections or components
        3. Important configuration or data
        Keep the summary concise (200-300 words max)."""
    
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Summarize this file ({file_path}):\n\n{content}"}
            ]
        )
        
        summary = completion.choices[0].message.content
        if not summary:
            return f"File: {file_path} (Content length: {len(content)} chars) - Summary generation failed"
            
        return f"### FILE SUMMARY: {file_path}\n{summary.strip()}"
        
    except Exception as e:
        print(f"Error summarizing file {file_path}: {e}")
        # Fallback to basic info
        return f"### FILE: {file_path}\nContent length: {len(content)} chars\nFile type: {file_extension or 'unknown'}"


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


async def run_docker_based_tests(project_path: Path, project_slug: str) -> Tuple[bool, str]:
    """
    Builds a Docker image to test the generated project.
    Returns (True, "Build successful") on success, or (False, error_logs) on failure.
    """
    dockerfile_path = project_path / "Dockerfile"
    if not dockerfile_path.exists():
        return False, "Dockerfile not found in the project root."

    image_tag = f"devbot-test-build-{project_slug}"
    logs = f"--- Building Docker image '{image_tag}' for project at {project_path} ---\n"

    try:
        build_cmd = ["docker", "build", "-t", image_tag, "."]

        process = await asyncio.create_subprocess_exec(
            *build_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(project_path)
        )
        stdout, stderr = await process.communicate()

        stdout_str = stdout.decode()
        stderr_str = stderr.decode()

        logs += stdout_str
        logs += stderr_str

        if process.returncode == 0:
            logs += "\n--- Docker build successful! ---"
            # Clean up the generated image to save space
            try:
                cleanup_cmd = ["docker", "rmi", image_tag]
                await asyncio.create_subprocess_exec(*cleanup_cmd)
            except Exception as e:
                logs += f"\nWarning: Failed to clean up Docker image {image_tag}: {e}"
            return True, logs
        else:
            logs += "\n--- Docker build failed. ---"
            return False, logs

    except FileNotFoundError:
        return False, "Docker command not found. Please ensure Docker is installed and in your PATH."
    except Exception as e:
        logs += f"\n--- An unexpected error occurred during Docker build: {e} ---\n"
        return False, logs


async def enhanced_debug_code_with_ai(project_path: Path, brd_data: models.BRDCreatePayload, 
                                     error_logs: str, debug_session: models.DebugSession,
                                     session: models.MultiAgentSession) -> List[str]:
    """
    Enhanced debugging with attempt tracking, previous attempt summarization, and multiple AI strategies.
    Returns list of modified file paths.
    """
    print(f"\n--- Enhanced AI Debugging - Attempt {debug_session.current_attempt + 1} ---")

    # Read all project files
    project_files = {}
    for root, _, files in os.walk(project_path):
        for file in files:
            if ".venv" in root or "__pycache__" in root or file == ".DS_Store":
                continue
            file_path = Path(root) / file
            relative_path = file_path.relative_to(project_path)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    project_files[str(relative_path)] = f.read()
            except Exception as e:
                print(f"Warning: Could not read file {file_path}: {e}")

    # Create summary of previous attempts
    previous_attempts_summary = ""
    if debug_session.attempts:
        previous_attempts_summary = "\n\n**Previous Debugging Attempts:**\n"
        for attempt in debug_session.attempts:
            previous_attempts_summary += f"- Attempt {attempt.attempt_number}: {attempt.strategy_used} -> {'SUCCESS' if attempt.success else 'FAILED'}\n"
            previous_attempts_summary += f"  Files modified: {', '.join(attempt.files_modified)}\n"
            if not attempt.success:
                previous_attempts_summary += f"  Error: {attempt.error_logs[:500]}...\n"

    # Prepare files content
    files_str = "\n\n".join([f"### `/{path}`\n\n```\n{content}\n```" for path, content in project_files.items()])

    # Enhanced debugging prompt with multiple strategies
    debug_strategies = [
        "dependency_fix",  # Focus on package/dependency issues
        "dockerfile_optimization",  # Focus on Dockerfile issues
        "code_compilation",  # Focus on source code compilation errors
        "configuration_fix",  # Focus on configuration files
        "multi_stage_analysis"  # Comprehensive analysis of all aspects
    ]

    current_strategy = debug_strategies[min(debug_session.current_attempt, len(debug_strategies) - 1)]

    strategy_prompts = {
        "dependency_fix": "Focus on fixing package dependencies, version conflicts, and package manager files (requirements.txt, package.json, go.mod, etc.)",
        "dockerfile_optimization": "Focus on optimizing the Dockerfile, fixing base image issues, and container build problems",
        "code_compilation": "Focus on fixing source code compilation errors, syntax errors, and import issues",
        "configuration_fix": "Focus on fixing configuration files, environment variables, and application settings",
        "multi_stage_analysis": "Perform a comprehensive analysis of all potential issues including dependencies, Dockerfile, source code, and configuration"
    }

    debug_prompt = f"""
    You are an expert debugging agent specializing in {current_strategy}. The Docker build for this generated project failed.
    
    **Current Strategy:** {strategy_prompts[current_strategy]}
    
    **Docker Build Error Logs:**
    ```
    {error_logs}
    ```
    
    {previous_attempts_summary}
    
    **Project BRD (for context):**
    {brd_data.model_dump_json(indent=2)}
    
    **Project Files:**
    {files_str}
    
    Based on your specialized analysis and the previous attempts, provide targeted file updates to fix the build.
    Be specific about what strategy you're applying and why it should work better than previous attempts.
    
    Return the result as a JSON object matching the `DebuggingUpdate` schema.
    """

    try:
        # Call specialized debugging agent
        updates_response, agent_call = await call_ai_agent(
            agent_type=f"debugger_{current_strategy}",
            prompt=debug_prompt,
            response_model=models.DebuggingUpdate,
            session=session
        )

        modified_files = []
        if updates_response.updates:
            for file_update in updates_response.updates:
                file_path = project_path / file_update.path
                print(f"Applying {current_strategy} fix to: {file_path}")
                if file_path.parent.exists():
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(file_update.updatedCode)
                    modified_files.append(file_update.path)
                else:
                    print(f"Warning: Parent directory for {file_path} does not exist. Skipping update.")

        # Record the attempt
        attempt = models.DebugAttempt(
            attempt_number=debug_session.current_attempt + 1,
            error_logs=error_logs,
            strategy_used=current_strategy,
            files_modified=modified_files,
            success=False  # Will be updated after testing
        )
        debug_session.attempts.append(attempt)
        debug_session.current_attempt += 1

        return modified_files

    except Exception as e:
        print(f"Enhanced debugging failed with exception: {e}")
        raise

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

    # Initialize multi-agent session for tracking
    session = models.MultiAgentSession(
        project_context=f"Repository generation for: {brd_data.projectName}",
        current_phase="generation"
    )

    # Step 1: Generate project file structure with dependencies using AI agent
    structure_prompt = f"""
    Based on the following Business Requirements Document (BRD), define a complete project file structure.
    For each file, specify its full path (relative to project root), a detailed description of its content/purpose,
    and an array of other file paths it depends on ('dependsOn').

    **Crucially, you MUST include a production-ready, multi-stage `Dockerfile`** to build and run the application.
    Also include all necessary package management files (e.g., `requirements.txt` for Python, `package.json` for Node.js, `go.mod` for Go, etc.).
    The file order must consider dependencies (e.g., `package.json` before `index.js`).

    Support for multiple frameworks and languages:
    - Backend: FastAPI, Actix (Rust), Gin (Go), Elysia (Bun/TypeScript)
    - ML/AI: PyTorch, Adalflow, Pydantic
    - Database: SQLAlchemy, other ORMs
    - Choose the most appropriate technology based on the BRD requirements

    BRD:
    {brd_data.model_dump_json(indent=2)}

    Return the file structure as a JSON object matching the DependencyOrder schema.
    """
    
    dependency_structure, _ = await call_ai_agent(
        agent_type="structure_generator",
        prompt=structure_prompt,
        response_model=models.DependencyOrder,
        session=session
    )
    
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

    # Step 3: Generate each file using specialized code generation agents
    generated_file_contents: Dict[str, str] = {}
    generated_file_summaries: Dict[str, str] = {}
    brd_string = brd_data.model_dump_json(indent=2)
    session.current_phase = "file_generation"

    for file_info in sorted_files:
        # Create intelligent summaries of all previously generated files for context
        context_parts = []
        for path, content in generated_file_contents.items():
            if path in generated_file_summaries:
                context_parts.append(generated_file_summaries[path])
            else:
                # Generate summary if not already created
                summary = await summarize_file_content(path, content)
                generated_file_summaries[path] = summary
                context_parts.append(summary)
        
        context_string = "\n\n".join(context_parts) if context_parts else "No relevant files have been created yet."

        file_gen_prompt = f"""
        Generate the complete and raw source code/content for the file: {file_info.path}
        File Description: {file_info.description}
        Overall Project BRD: {brd_string}
        
        Previously Generated Files Context (Summaries):
        {context_string}
        
        Based on the file summaries above, ensure your generated code:
        - Integrates properly with existing files and their functionality
        - Uses consistent naming conventions and patterns
        - Imports and references the correct modules/functions from other files
        - Follows the established architecture and design patterns
        - Maintains consistency with the technology stack and dependencies
        
        Code Quality Guidelines:
        - Follow language-specific conventions
        - Include proper error handling
        - Add necessary imports and dependencies
        - Use production-ready patterns
        - Include basic logging where appropriate
        
        Important: Output ONLY the raw content for '{file_info.path}'. Do not include explanations or markdown fences.
        """
        
        print(f"Generating content for {file_info.path}...")
        print(f"  Context includes summaries of {len(context_parts)} previously generated files")
        file_content, _ = await call_ai_agent(
            agent_type="code_generator",
            prompt=file_gen_prompt,
            session=session
        )

        full_file_path = project_path / file_info.path
        full_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_file_path, "w", encoding="utf-8") as f:
            f.write(file_content)
        print(f"Successfully created file: {full_file_path}")
        generated_file_contents[file_info.path] = file_content

    # Step 4: Enhanced Debug Loop with Attempt Tracking
    session.current_phase = "debugging"
    debug_session = models.DebugSession(
        project_path=str(project_path),
        max_attempts=5
    )

    while debug_session.current_attempt < debug_session.max_attempts:
        attempt_num = debug_session.current_attempt + 1
        print(f"\n--- Docker Build & Debug Cycle: Attempt {attempt_num}/{debug_session.max_attempts} ---")

        success, logs = await run_docker_based_tests(project_path, project_name_slug)
        print(logs)

        if success:
            # Mark the last attempt as successful if there were any
            if debug_session.attempts:
                debug_session.attempts[-1].success = True
            
            print("\n--- Docker build successful! Moving to validation phase. ---")
            break

        print("\n--- Docker build failed. Initiating enhanced AI debugging... ---")
        
        if debug_session.current_attempt < debug_session.max_attempts - 1:
            try:
                modified_files = await enhanced_debug_code_with_ai(
                    project_path, brd_data, logs, debug_session, session
                )
                print(f"Modified {len(modified_files)} files in this debug attempt")
            except Exception as e:
                print(f"AI debugging failed with an exception: {e}. Retrying...")
        else:
            print("\n--- Maximum debugging retries reached. ---")
            # Generate final debugging report
            print("\n--- Debug Session Summary ---")
            for attempt in debug_session.attempts:
                print(f"Attempt {attempt.attempt_number}: {attempt.strategy_used} -> {'SUCCESS' if attempt.success else 'FAILED'}")
            raise Exception("Failed to generate a working project after multiple Docker build attempts.")

    # Step 5: Validation Phase with AI Agent
    session.current_phase = "validation"
    print("\n--- Starting Validation Phase ---")
    
    try:
        # Create validation plan
        validation_plan = await create_validation_plan(project_path, brd_data, session)
        print(f"Created validation plan with {len(validation_plan.tests)} tests")
        
        # Run validation tests
        validation_report = await run_validation_tests(project_path, validation_plan, session)
        
        print(f"\n--- Validation Results ---")
        print(f"Overall Success: {'✅' if validation_report.overall_success else '❌'}")
        
        for result in validation_report.results:
            status = "✅" if result.success else "❌"
            print(f"{status} {result.test_type}")
            if not result.success and result.error_message:
                print(f"  Error: {result.error_message}")
        
        if validation_report.recommendations:
            print("\n--- Recommendations ---")
            for rec in validation_report.recommendations:
                print(f"• {rec}")
        
        # Save validation report
        report_path = project_path / "validation_report.json"
        with open(report_path, "w") as f:
            f.write(validation_report.model_dump_json(indent=2))
        
        print(f"\n--- Repository Generation Complete ---")
        print(f"Project Path: {project_path}")
        print(f"Validation Report: {report_path}")
        print(f"Total AI Agent Calls: {len(session.agent_calls)}")
        
        return str(project_path)
        
    except Exception as e:
        print(f"Validation phase failed: {e}")
        print("Project generated successfully but validation could not be completed.")
        return str(project_path)

async def call_ai_agent(agent_type: str, prompt: str, response_model: Optional[Type[T]] = None, 
                       session: Optional[models.MultiAgentSession] = None, model_name: str = config.DEFAULT_MODEL) -> Tuple[T, models.AIAgentCall]:
    """
    Enhanced AI agent call with tracking and session management.
    """
    try:
        if response_model:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": f"You are a specialized {agent_type} agent. Please respond in JSON format matching the following Pydantic schema: {response_model.model_json_schema()}"},
                    {"role": "user", "content": prompt}
                ]
            )
            response_content = completion.choices[0].message.content
            parsed_content = json.loads(response_content)
            validated_response = response_model.model_validate(parsed_content)
        else:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": f"You are a specialized {agent_type} agent."},
                    {"role": "user", "content": prompt}
                ]
            )
            response_content = completion.choices[0].message.content
            validated_response = response_content

        # Create agent call record
        agent_call = models.AIAgentCall(
            agent_type=agent_type,
            prompt=prompt,
            response=str(response_content),
            model_used=model_name
        )

        # Add to session if provided
        if session:
            session.agent_calls.append(agent_call)

        return validated_response, agent_call

    except Exception as e:
        print(f"Error in AI agent call ({agent_type}): {e}")
        raise

async def create_validation_plan(project_path: Path, brd_data: models.BRDCreatePayload, 
                                session: models.MultiAgentSession) -> models.ValidationPlan:
    """
    Creates a comprehensive validation plan based on the project type and BRD requirements.
    """
    print("\n--- Creating Validation Plan ---")

    # Read project files to understand structure
    project_files = {}
    for root, _, files in os.walk(project_path):
        for file in files:
            if ".venv" in root or "__pycache__" in root or file == ".DS_Store":
                continue
            file_path = Path(root) / file
            relative_path = file_path.relative_to(project_path)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    project_files[str(relative_path)] = f.read()
            except Exception as e:
                print(f"Warning: Could not read file {file_path}: {e}")

    files_str = "\n\n".join([f"### `/{path}`\n\n```\n{content[:1000]}...\n```" 
                           for path, content in project_files.items()])

    validation_prompt = f"""
    You are a validation planning agent. Create a comprehensive test plan for this generated project.
    
    **Project BRD:**
    {brd_data.model_dump_json(indent=2)}
    
    **Project Structure:**
    {files_str}
    
    Based on the project type, technology stack, and features, create validation tests that include:
    1. Build/compilation verification
    2. Basic functionality tests
    3. API endpoint tests (if applicable)
    4. Database connectivity tests (if applicable)
    5. Integration tests for core features
    
    Consider the technology stack and create appropriate test commands. For example:
    - For Python/FastAPI: pytest commands, curl tests for endpoints
    - For Node.js: npm test, endpoint testing
    - For Go: go test, build verification
    - For Docker: container health checks
    
    Return a comprehensive validation plan as JSON matching the ValidationPlan schema.
    """

    validation_plan, agent_call = await call_ai_agent(
        agent_type="validation_planner",
        prompt=validation_prompt,
        response_model=models.ValidationPlan,
        session=session
    )

    return validation_plan


async def run_validation_tests(project_path: Path, validation_plan: models.ValidationPlan,
                              session: models.MultiAgentSession) -> models.ValidationReport:
    """
    Executes the validation plan and generates a comprehensive report.
    """
    print("\n--- Running Validation Tests ---")
    results = []

    # Run setup commands
    for setup_cmd in validation_plan.setup_commands:
        print(f"Setup: {setup_cmd}")
        try:
            process = await asyncio.create_subprocess_shell(
                setup_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(project_path)
            )
            stdout, stderr = await process.communicate()
            if process.returncode != 0:
                print(f"Setup command failed: {stderr.decode()}")
        except Exception as e:
            print(f"Setup command error: {e}")

    # Run each validation test
    for test in validation_plan.tests:
        print(f"Running {test.test_type} test: {test.test_command}")
        
        try:
            process = await asyncio.create_subprocess_shell(
                test.test_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(project_path)
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), 
                    timeout=test.timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                results.append(models.ValidationResult(
                    test_type=test.test_type,
                    success=False,
                    output="",
                    error_message=f"Test timed out after {test.timeout} seconds"
                ))
                continue

            stdout_str = stdout.decode()
            stderr_str = stderr.decode()
            output = stdout_str + stderr_str

            success = process.returncode == 0
            if test.expected_output and success:
                success = test.expected_output in output

            results.append(models.ValidationResult(
                test_type=test.test_type,
                success=success,
                output=output,
                error_message=stderr_str if not success else None
            ))

        except Exception as e:
            results.append(models.ValidationResult(
                test_type=test.test_type,
                success=False,
                output="",
                error_message=str(e)
            ))

    # Run cleanup commands
    for cleanup_cmd in validation_plan.cleanup_commands:
        print(f"Cleanup: {cleanup_cmd}")
        try:
            process = await asyncio.create_subprocess_shell(
                cleanup_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(project_path)
            )
            await process.communicate()
        except Exception as e:
            print(f"Cleanup command error: {e}")

    # Analyze results and generate recommendations
    overall_success = all(result.success for result in results)
    
    # Generate recommendations using AI
    failed_tests = [r for r in results if not r.success]
    recommendations = []
    
    if failed_tests:
        recommendations_prompt = f"""
        The following validation tests failed for the generated project:
        
        {json.dumps([{"test_type": r.test_type, "error": r.error_message, "output": r.output} for r in failed_tests], indent=2)}
        
        Provide specific recommendations to fix these issues. Focus on actionable steps.
        Return as a JSON list of recommendation strings.
        """
        
        try:
            recs_response, agent_call = await call_ai_agent(
                agent_type="recommendations_generator",
                prompt=recommendations_prompt,
                session=session
            )
            if isinstance(recs_response, str):
                recommendations = json.loads(recs_response)
            else:
                recommendations = ["Manual review required for failed tests"]
        except Exception as e:
            print(f"Failed to generate recommendations: {e}")
            recommendations = ["Manual review required for failed tests"]

    return models.ValidationReport(
        overall_success=overall_success,
        results=results,
        recommendations=recommendations
    )
