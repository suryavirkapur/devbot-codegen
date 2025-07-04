import os
import shutil
import subprocess
import asyncio
import json
import zipfile
from datetime import datetime

from pathlib import Path
from typing import TypeVar, Type, List, Dict, Optional, Tuple
from openai import OpenAI, AzureOpenAI
from . import config
from . import models
from .advanced_debugging import (
    LanguageDetector, ErrorAnalyzer, TestStrategyManager, 
    ParallelTestExecutor, LearningSystem, CheckpointManager, QualityAssessor
)

if os.getenv("AZURE_OPENAI_ENDPOINT"): # Example condition to use Azure
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT") or ""
    )

else:
    client = OpenAI(api_key=config.OPENAI_API_KEY, base_url="http://0.0.0.0:11437/v1")


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

async def get_embedding(text: str, model: str = "nomic-embed-text") -> List[float]:
    """
    Generates an embedding for a given text using the configured OpenAI client.
    """
    try:
        # Normalize newlines to avoid issues with some models
        text = text.replace("\n", " ")
        response = client.embeddings.create(
            input=[text],
            model=model
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding for text: '{text[:100]}...': {e}")
        raise

async def summarize_code_chunk(chunk_text: str, file_path: str, language: str, model_name: str = config.DEFAULT_MODEL) -> str:
    """
    Generates a summary for a code chunk using an LLM.
    """
    prompt = f"""
    Please provide a concise, one-sentence summary of the following code snippet.
    The snippet is from the file '{file_path}' and is written in {language}.
    Focus on the primary purpose or action of the code.

    Code Snippet:
    ```
    {chunk_text}
    ```

    One-sentence summary:
    """
    try:
        # Re-using generate_code_from_prompt as it's a simple text generation task
        summary = await generate_code_from_prompt(prompt, model_name=model_name)
        return summary.strip()
    except Exception as e:
        print(f"Error summarizing chunk from {file_path}: {e}")
        return "Could not generate summary."

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


async def advanced_debug_with_ai(project_path: Path, brd_data: models.BRDCreatePayload,
                                error_logs: str, advanced_debug_session: models.AdvancedDebugSession,
                                session: models.MultiAgentSession) -> List[str]:
    """
    Advanced AI debugging with language detection, error classification, parallel testing, and learning.
    Returns list of modified file paths.
    """
    print(f"\n--- Advanced AI Debugging - Attempt {advanced_debug_session.current_attempt + 1} ---")

    # Initialize systems
    learning_system = LearningSystem()
    checkpoint_manager = CheckpointManager(project_path)
    
    # Create checkpoint before debugging
    checkpoint = await checkpoint_manager.create_checkpoint(
        f"Before debug attempt {advanced_debug_session.current_attempt + 1}"
    )

    # Detect technology stack if not already done
    if not advanced_debug_session.tech_stack:
        print("Detecting technology stack...")
        tech_stack = await LanguageDetector.detect_tech_stack(project_path)
        advanced_debug_session.tech_stack = tech_stack
        print(f"Detected: {tech_stack.primary_language.language} with frameworks: {[f.name for f in tech_stack.frameworks]}")
    else:
        tech_stack = advanced_debug_session.tech_stack

    # Analyze errors
    print("Analyzing errors...")
    error_analyses = await ErrorAnalyzer.analyze_errors(error_logs, tech_stack)
    
    # Find applicable learning patterns
    applicable_patterns = []
    for error_analysis in error_analyses:
        patterns = learning_system.find_applicable_patterns(error_analysis, tech_stack)
        applicable_patterns.extend(patterns)
    
    print(f"Found {len(applicable_patterns)} applicable learning patterns")

    # Get testing strategies
    strategies = TestStrategyManager.get_language_strategies(tech_stack)
    
    # Create parallel execution plan
    execution_plan = models.ParallelTestExecution(
        strategies=strategies[:5],  # Use top 5 strategies
        max_parallel=3,
        timeout=300
    )

    # Execute strategies in parallel
    print(f"Executing {len(execution_plan.strategies)} strategies in parallel...")
    test_results = await ParallelTestExecutor.execute_parallel_strategies(
        project_path, execution_plan, tech_stack
    )

    # Find the best result
    successful_results = [r for r in test_results if r.success]
    best_result = None
    
    if successful_results:
        # Sort by confidence score
        best_result = max(successful_results, key=lambda r: r.confidence_score)
        print(f"Best successful strategy: {best_result.strategy_id} (confidence: {best_result.confidence_score:.2f})")
        
        # Mark checkpoint as working if we have a successful result
        if best_result.confidence_score > 0.7:
            await checkpoint_manager.create_checkpoint(
                f"Working state after successful {best_result.strategy_id}",
                working_state=True
            )
    else:
        # No successful strategies, try AI-based fixes
        print("No strategies succeeded, applying AI-based fixes...")
        
        # Apply patterns from learning system
        modified_files = []
        for pattern in applicable_patterns[:3]:  # Try top 3 patterns
            try:
                for change in pattern.code_changes:
                    if 'file_path' in change and 'content' in change:
                        file_path = project_path / change['file_path']
                        if file_path.exists():
                            with open(file_path, 'w', encoding='utf-8') as f:
                                f.write(change['content'])
                            modified_files.append(change['file_path'])
                            print(f"Applied learning pattern to {change['file_path']}")
                
                # Test if this pattern worked
                success, test_output = await run_docker_based_tests(
                    project_path, project_path.name
                )
                
                if success:
                    learning_system.learn_from_success(error_analyses[0], [change], tech_stack)
                    best_result = models.TestResult(
                        strategy_id=f"learning_pattern_{pattern.pattern_id}",
                        success=True,
                        execution_time=0,
                        output=test_output,
                        files_modified=modified_files,
                        confidence_score=0.8
                    )
                    break
                else:
                    learning_system.learn_from_failure(pattern.pattern_id)
                    # Rollback changes
                    await checkpoint_manager.rollback_to_checkpoint(checkpoint.checkpoint_id)
                    
            except Exception as e:
                print(f"Error applying pattern {pattern.pattern_id}: {e}")
                learning_system.learn_from_failure(pattern.pattern_id)

    # Record the debug attempt
    attempt = models.AdvancedDebugAttempt(
        attempt_number=advanced_debug_session.current_attempt + 1,
        tech_stack=tech_stack,
        error_analysis=error_analyses,
        strategies_attempted=test_results,
        best_result=best_result,
        checkpoint_created=checkpoint.checkpoint_id,
        learning_applied=[p.pattern_id for p in applicable_patterns],
        success=best_result is not None and best_result.success
    )
    
    advanced_debug_session.attempts.append(attempt)
    advanced_debug_session.current_attempt += 1
    
    # Save debug session state
    try:
        session_data = advanced_debug_session.model_dump()
        with open(project_path / "advanced_debug_session.json", "w") as f:
            json.dump(session_data, f, indent=2, default=str)
    except Exception as e:
        print(f"Warning: Could not save debug session: {e}")

    if best_result and best_result.success:
        print(f"Debug attempt successful with strategy: {best_result.strategy_id}")
        
        # Learn from successful attempt if it was an AI-generated fix
        if error_analyses and not best_result.strategy_id.startswith("learning_pattern"):
            changes = [{'strategy': best_result.strategy_id, 'files': best_result.files_modified}]
            learning_system.learn_from_success(error_analyses[0], changes, tech_stack)
        
        return best_result.files_modified
    else:
        print("Debug attempt failed - no successful strategies found")
        return []

async def generate_repository_files(brd_data: models.BRDCreatePayload, session: models.MultiAgentSession) -> str:
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

    # Update session context
    session.project_context = f"Repository generation for: {brd_data.projectName}"
    session.current_phase="generation"

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

    # Step 4: Advanced Debug Loop with Multi-Strategy Testing and Learning
    session.current_phase = "debugging"
    
    # Initialize advanced debug session
    checkpoint_manager = CheckpointManager(project_path)
    initial_checkpoint = await checkpoint_manager.create_checkpoint("Initial project state", working_state=False)
    
    advanced_debug_session = models.AdvancedDebugSession(
        project_path=str(project_path),
        rollback_manager=checkpoint_manager.rollback_manager,
        learning_session=models.LearningSession(project_context=f"Project: {brd_data.projectName}"),
        max_attempts=10  # Increased attempts due to more sophisticated debugging
    )

    while advanced_debug_session.current_attempt < advanced_debug_session.max_attempts:
        attempt_num = advanced_debug_session.current_attempt + 1
        print(f"\n--- Advanced Docker Build & Debug Cycle: Attempt {attempt_num}/{advanced_debug_session.max_attempts} ---")

        success, logs = await run_docker_based_tests(project_path, project_name_slug)
        print(logs)

        if success:
            # Mark the last attempt as successful if there were any
            if advanced_debug_session.attempts:
                advanced_debug_session.attempts[-1].success = True

            # Create final working checkpoint
            await checkpoint_manager.create_checkpoint("Final working state", working_state=True)
            
            print("\n--- Docker build successful! Moving to validation phase. ---")
            break

        print("\n--- Docker build failed. Initiating advanced AI debugging... ---")

        if advanced_debug_session.current_attempt < advanced_debug_session.max_attempts - 1:
            try:
                modified_files = await advanced_debug_with_ai(
                    project_path, brd_data, logs, advanced_debug_session, session
                )
                print(f"Modified {len(modified_files)} files in this debug attempt")
                
                # If no files were modified and we have working checkpoints, try rolling back
                if not modified_files:
                    latest_working = checkpoint_manager.get_latest_working_checkpoint()
                    if latest_working:
                        print("No modifications applied, rolling back to latest working state...")
                        await checkpoint_manager.rollback_to_checkpoint(latest_working.checkpoint_id)
                        
            except Exception as e:
                print(f"Advanced AI debugging failed with an exception: {e}. Retrying...")
                # Try to rollback to a working state if available
                latest_working = checkpoint_manager.get_latest_working_checkpoint()
                if latest_working:
                    print("Rolling back to latest working checkpoint due to debugging error...")
                    await checkpoint_manager.rollback_to_checkpoint(latest_working.checkpoint_id)
        else:
            print("\n--- Maximum debugging retries reached. ---")
            # Generate comprehensive debugging report
            print("\n--- Advanced Debug Session Summary ---")
            for attempt in advanced_debug_session.attempts:
                print(f"Attempt {attempt.attempt_number}: {len(attempt.strategies_attempted)} strategies -> {'SUCCESS' if attempt.success else 'FAILED'}")
                if attempt.best_result:
                    print(f"  Best strategy: {attempt.best_result.strategy_id} (confidence: {attempt.best_result.confidence_score:.2f})")
                print(f"  Learning patterns applied: {len(attempt.learning_applied)}")
            
            # Try one final rollback to the latest working state
            latest_working = checkpoint_manager.get_latest_working_checkpoint()
            if latest_working:
                print(f"Final attempt: Rolling back to working checkpoint: {latest_working.description}")
                await checkpoint_manager.rollback_to_checkpoint(latest_working.checkpoint_id)
                
                # Test one more time
                success, logs = await run_docker_based_tests(project_path, project_name_slug)
                if success:
                    print("Rollback to working state successful!")
                    break
            
            raise Exception("Failed to generate a working project after advanced debugging attempts.")

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

    files_str = "\n\n".join([f"### `/{path}`\n\n```\n{content[:2000]}...\n```"
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

def create_project_zip(project_path: Path, zip_name: str) -> str:
    """
    Creates a zip file of the generated project.
    Returns the path to the created zip file.
    """
    zip_path = project_path.parent / f"{zip_name}.zip"

    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(project_path):
                for file in files:
                    file_path = Path(root) / file
                    # Calculate relative path from project root
                    relative_path = file_path.relative_to(project_path)
                    zipf.write(file_path, relative_path)

        print(f"Created zip file: {zip_path}")
        return str(zip_path)

    except Exception as e:
        print(f"Error creating zip file: {e}")
        raise

async def unified_project_generation(request: models.UnifiedProjectRequest) -> models.ProjectGenerationResult:
    """
    Unified service that handles the entire project generation flow:
    1. Generate BRD from business requirements
    2. Generate project files
    3. Validate and debug
    4. Create zip file
    5. Return comprehensive results
    """
    start_time = datetime.utcnow()
    session = models.MultiAgentSession(
        project_context=f"Initial context for {request.projectName or 'new project'}",
    )
    project_path = None


    try:
        # Step 1: Generate BRD from business requirements
        print("=== Step 1: Generating BRD from business requirements ===")
        brd_data = await generate_brd_from_text_with_ai(request.businessRequirement)

        # Override project name if provided
        if request.projectName:
            brd_data.projectName = request.projectName

        # Add additional instructions if provided
        if request.additionalInstructions:
            current_additional = brd_data.additionalRequirements or ""
            brd_data.additionalRequirements = f"{current_additional}\n\nAdditional Instructions: {request.additionalInstructions}".strip()

        print(f"Generated BRD for project: {brd_data.projectName}")

        # Step 2: Improve BRD with AI
        print("=== Step 2: Improving BRD with AI ===")
        improved_brd = await improve_brd_with_ai(brd_data)

        # Step 3: Generate repository files
        print("=== Step 3: Generating repository files ===")
        project_path = await generate_repository_files(improved_brd, session)
        project_path_obj = Path(project_path)

        # Step 4: Count generated files
        total_files = sum(1 for _ in project_path_obj.rglob('*') if _.is_file())

        # Step 5: Load validation report if exists
        validation_report = None
        validation_report_path = project_path_obj / "validation_report.json"
        if validation_report_path.exists():
            try:
                with open(validation_report_path, 'r') as f:
                    validation_data = json.loads(f.read())
                    validation_report = models.ValidationReport.model_validate(validation_data)
            except Exception as e:
                print(f"Could not load validation report: {e}")

        # Step 6: Create zip file
        print("=== Step 4: Creating zip file ===")
        project_name_slug = improved_brd.projectName.lower().replace(" ", "-").replace(r"[^a-z0-9-]", "")
        zip_path = create_project_zip(project_path_obj, f"{project_name_slug}-generated")

        # Step 7: Advanced Quality Assessment
        print("=== Step 7: Performing advanced quality assessment ===")
        
        # Detect tech stack for final assessment
        tech_stack_detected = await LanguageDetector.detect_tech_stack(project_path_obj)
        
        # Assess project health
        project_health = await QualityAssessor.assess_project_health(project_path_obj, tech_stack_detected)
        
        # Step 8: Calculate metrics
        end_time = datetime.utcnow()
        generation_time = (end_time - start_time).total_seconds()

        # Extract advanced debug session data
        debug_attempts = 0
        advanced_debug_session_data = None
        advanced_debug_session_path = project_path_obj / "advanced_debug_session.json"
        if advanced_debug_session_path.exists():
            try:
                with open(advanced_debug_session_path, 'r') as f:
                    debug_data = json.loads(f.read())
                    debug_attempts = len(debug_data.get("attempts", []))
                    advanced_debug_session_data = models.AdvancedDebugSession.model_validate(debug_data)
            except Exception as e:
                print(f"Warning: Could not load advanced debug session: {e}")

        # Extract AI agent calls count
        ai_calls_count = len(session.agent_calls) if session else 0

        print(f"=== Advanced Project Generation Completed Successfully ===")
        print(f"Project: {improved_brd.projectName}")
        print(f"Primary Language: {tech_stack_detected.primary_language.language}")
        print(f"Frameworks: {[f.name for f in tech_stack_detected.frameworks]}")
        print(f"Files generated: {total_files}")
        print(f"Generation time: {generation_time:.2f} seconds")
        print(f"Debug attempts: {debug_attempts}")
        print(f"Project health score: {project_health.overall_health:.1f}/100")
        print(f"Build success: {'✅' if project_health.build_success else '❌'}")
        print(f"Tests passing: {'✅' if project_health.tests_passing else '❌'}")
        print(f"Zip file: {zip_path}")
        
        if project_health.recommendations:
            print("Recommendations for improvement:")
            for rec in project_health.recommendations[:3]:  # Show top 3
                print(f"  • {rec}")

        return models.ProjectGenerationResult(
            success=True,
            project_name=improved_brd.projectName,
            brd_data=improved_brd,
            zip_file_path=zip_path,
            generation_time_seconds=generation_time,
            validation_report=validation_report,
            debug_attempts=debug_attempts,
            total_files_generated=total_files,
            ai_agent_calls=ai_calls_count,
            tech_stack_detected=tech_stack_detected,
            project_health=project_health,
            advanced_debug_session=advanced_debug_session_data
        )

    except Exception as e:
        end_time = datetime.utcnow()
        generation_time = (end_time - start_time).total_seconds()

        print(f"Error in unified project generation: {e}")

        # Determine which stage failed
        stage = "brd_generation"
        if 'brd_data' in locals():
            stage = "repository_generation"
        if 'project_path' in locals():
            stage = "validation"
        if 'zip_path' in locals():
            stage = "finalization"

        raise Exception(f"Project generation failed at {stage}: {str(e)}")
