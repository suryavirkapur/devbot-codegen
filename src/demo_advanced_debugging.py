#!/usr/bin/env python3
"""
Advanced Debugging System Demonstration

This script demonstrates the capabilities of the enhanced debugging system including:
- Language and framework detection
- Error classification and analysis
- Multi-strategy parallel testing
- Progressive learning from fixes
- Quality assessment and health monitoring
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import models
from advanced_debugging import (
    LanguageDetector, ErrorAnalyzer, TestStrategyManager,
    ParallelTestExecutor, LearningSystem, CheckpointManager, QualityAssessor
)


async def demo_language_detection():
    """Demonstrate advanced language and framework detection"""
    print("🔍 === Language Detection Demo ===")
    
    # Create a sample project structure
    demo_project = Path("demo_project")
    demo_project.mkdir(exist_ok=True)
    
    # Create sample files for different languages/frameworks
    sample_files = {
        "package.json": {
            "name": "demo-app",
            "dependencies": {
                "express": "^4.18.0",
                "react": "^18.0.0"
            }
        },
        "main.py": """
from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}
""",
        "Dockerfile": """
FROM python:3.9
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "main.py"]
""",
        "requirements.txt": """
fastapi==0.68.0
uvicorn==0.15.0
""",
        "go.mod": """
module example.com/demo
go 1.19
require github.com/gin-gonic/gin v1.8.0
"""
    }
    
    # Write sample files
    for filename, content in sample_files.items():
        file_path = demo_project / filename
        if isinstance(content, dict):
            with open(file_path, 'w') as f:
                json.dump(content, f, indent=2)
        else:
            with open(file_path, 'w') as f:
                f.write(content)
    
    # Detect technology stack
    tech_stack = await LanguageDetector.detect_tech_stack(demo_project)
    
    print(f"Primary Language: {tech_stack.primary_language.language} (confidence: {tech_stack.primary_language.confidence:.2f})")
    
    if tech_stack.secondary_languages:
        print("Secondary Languages:")
        for lang in tech_stack.secondary_languages:
            print(f"  - {lang.language} (confidence: {lang.confidence:.2f})")
    
    print("Frameworks:")
    for framework in tech_stack.frameworks:
        print(f"  - {framework.name} ({framework.category}) (confidence: {framework.confidence:.2f})")
    
    print(f"Package Managers: {', '.join(tech_stack.package_managers)}")
    print(f"Build Tools: {', '.join(tech_stack.build_tools)}")
    print(f"Deployment Tools: {', '.join(tech_stack.deployment_tools)}")
    print(f"Databases: {', '.join(tech_stack.databases)}")
    
    # Cleanup
    import shutil
    shutil.rmtree(demo_project)
    
    return tech_stack


async def demo_error_analysis():
    """Demonstrate error classification and analysis"""
    print("\n🔧 === Error Analysis Demo ===")
    
    # Sample error logs from different scenarios
    sample_errors = [
        """
        ERROR: Could not find a version that satisfies the requirement fastapi==999.0.0
        No matching distribution found for fastapi==999.0.0
        """,
        """
        SyntaxError: invalid syntax
        File "main.py", line 15
            if condition
                      ^
        SyntaxError: invalid syntax
        """,
        """
        ModuleNotFoundError: No module named 'requests'
        File "/app/main.py", line 3, in <module>
            import requests
        """,
        """
        docker: Error response from daemon: driver failed programming external connectivity
        on endpoint test_container: Error starting userland proxy: listen tcp 0.0.0.0:8000:
        bind: address already in use.
        """
    ]
    
    # Create mock tech stack
    tech_stack = models.TechStackDetection(
        primary_language=models.LanguageInfo(language="python", confidence=0.95),
        frameworks=[models.FrameworkInfo(name="fastapi", category="web", confidence=0.8)]
    )
    
    for i, error_log in enumerate(sample_errors, 1):
        print(f"\n--- Error Analysis {i} ---")
        print(f"Error Log: {error_log.strip()}")
        
        analyses = await ErrorAnalyzer.analyze_errors(error_log, tech_stack)
        
        for analysis in analyses:
            print(f"Category: {analysis.classification.category}")
            print(f"Subcategory: {analysis.classification.subcategory}")
            print(f"Severity: {analysis.classification.severity}")
            print(f"Confidence: {analysis.classification.confidence:.2f}")
            
            if analysis.affected_files:
                print(f"Affected Files: {', '.join(analysis.affected_files)}")
            
            print("Suggested Fixes:")
            for fix in analysis.suggested_fixes[:2]:  # Show top 2
                print(f"  - {fix}")
            
            if analysis.root_cause:
                print(f"Root Cause: {analysis.root_cause}")


async def demo_parallel_testing():
    """Demonstrate multi-strategy parallel testing"""
    print("\n🚀 === Parallel Testing Demo ===")
    
    # Mock tech stack for Python/FastAPI
    tech_stack = models.TechStackDetection(
        primary_language=models.LanguageInfo(language="python", confidence=0.95),
        frameworks=[models.FrameworkInfo(name="fastapi", category="web", confidence=0.8)],
        package_managers=["pip"]
    )
    
    # Get applicable strategies
    strategies = TestStrategyManager.get_language_strategies(tech_stack)
    
    print(f"Available strategies for {tech_stack.primary_language.language}:")
    for strategy in strategies:
        print(f"  - {strategy.name} (Priority: {strategy.priority}, Success Rate: {strategy.success_rate:.2f})")
    
    # Create execution plan
    execution_plan = models.ParallelTestExecution(
        strategies=strategies[:3],  # Use top 3 strategies
        max_parallel=2,
        timeout=60
    )
    
    print(f"\nExecuting {len(execution_plan.strategies)} strategies in parallel...")
    
    # Note: This would fail in demo since we don't have a real project
    # In a real scenario, this would test actual project files
    print("(In a real scenario, this would execute actual build/test commands)")
    
    # Mock results for demonstration
    mock_results = [
        models.TestResult(
            strategy_id="build_validation",
            success=True,
            execution_time=15.5,
            output="Build completed successfully",
            confidence_score=0.9
        ),
        models.TestResult(
            strategy_id="python_deps",
            success=False,
            execution_time=8.2,
            output="Package installation failed",
            error_message="Could not install fastapi==999.0.0",
            confidence_score=0.0
        ),
        models.TestResult(
            strategy_id="syntax_check",
            success=True,
            execution_time=2.1,
            output="No syntax errors found",
            confidence_score=0.85
        )
    ]
    
    print("\nMock Test Results:")
    for result in mock_results:
        status = "✅ PASS" if result.success else "❌ FAIL"
        print(f"  {status} {result.strategy_id} ({result.execution_time:.1f}s, confidence: {result.confidence_score:.2f})")
        if result.error_message:
            print(f"    Error: {result.error_message}")


async def demo_learning_system():
    """Demonstrate progressive learning capabilities"""
    print("\n🧠 === Learning System Demo ===")
    
    learning_system = LearningSystem()
    
    # Create mock error analysis
    error_analysis = models.ErrorAnalysis(
        original_error="ModuleNotFoundError: No module named 'fastapi'",
        classification=models.ErrorCategory(
            category="dependency",
            subcategory="missing_package",
            severity="high",
            confidence=0.9
        ),
        affected_files=["main.py"],
        suggested_fixes=["Install fastapi package", "Add fastapi to requirements.txt"],
        root_cause="Missing dependency in requirements.txt"
    )
    
    # Mock tech stack
    tech_stack = models.TechStackDetection(
        primary_language=models.LanguageInfo(language="python", confidence=0.95),
        frameworks=[models.FrameworkInfo(name="fastapi", category="web", confidence=0.8)]
    )
    
    # Simulate learning from a successful fix
    successful_changes = [
        {
            "file_path": "requirements.txt",
            "change_type": "addition",
            "content": "fastapi==0.68.0\nuvicorn==0.15.0"
        }
    ]
    
    pattern = learning_system.learn_from_success(error_analysis, successful_changes, tech_stack)
    print(f"Learned new pattern: {pattern.pattern_id}")
    print(f"Pattern description: {pattern.fix_description}")
    print(f"Success count: {pattern.success_count}")
    
    # Find applicable patterns for similar error
    similar_error = models.ErrorAnalysis(
        original_error="ModuleNotFoundError: No module named 'uvicorn'",
        classification=models.ErrorCategory(
            category="dependency",
            subcategory="missing_package", 
            severity="high",
            confidence=0.9
        ),
        affected_files=["main.py"]
    )
    
    applicable_patterns = learning_system.find_applicable_patterns(similar_error, tech_stack)
    print(f"\nFound {len(applicable_patterns)} applicable patterns for similar error")
    
    for pattern in applicable_patterns:
        print(f"  - Pattern {pattern.pattern_id}: {pattern.fix_description}")
        print(f"    Success rate: {pattern.success_count}/{pattern.success_count + pattern.failure_count}")


async def demo_quality_assessment():
    """Demonstrate project quality assessment"""
    print("\n📊 === Quality Assessment Demo ===")
    
    # Create a temporary project for assessment
    demo_project = Path("quality_demo")
    demo_project.mkdir(exist_ok=True)
    
    # Create sample files
    (demo_project / "main.py").write_text("""
import os
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    # TODO: Add proper error handling
    return {"message": "Hello World"}

@app.get("/users/{user_id}")
def get_user(user_id: int):
    # Potential security issue: no input validation
    return {"user_id": user_id}

def complex_function():
    # This function is overly complex
    for i in range(100):
        for j in range(100):
            for k in range(100):
                if i + j + k > 150:
                    pass
    return True
""")

    (demo_project / "requirements.txt").write_text("""
fastapi==0.68.0
uvicorn==0.15.0
""")

    (demo_project / "Dockerfile").write_text("""
FROM python:3.9
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
""")

    # Create tech stack
    tech_stack = models.TechStackDetection(
        primary_language=models.LanguageInfo(language="python", confidence=0.95),
        frameworks=[models.FrameworkInfo(name="fastapi", category="web", confidence=0.8)],
        package_managers=["pip"],
        deployment_tools=["docker"]
    )
    
    # Assess project health
    project_health = await QualityAssessor.assess_project_health(demo_project, tech_stack)
    
    print(f"Overall Health Score: {project_health.overall_health:.1f}/100")
    print(f"Build Success: {'✅' if project_health.build_success else '❌'}")
    print(f"Tests Passing: {'✅' if project_health.tests_passing else '❌'}")
    
    print("\nQuality Metrics:")
    metrics = project_health.quality_metrics
    print(f"  Complexity Score: {metrics.complexity_score:.1f}/100")
    print(f"  Maintainability: {metrics.maintainability_score:.1f}/100")
    print(f"  Security Score: {metrics.security_score:.1f}/100")
    print(f"  Performance Score: {metrics.performance_score:.1f}/100")
    print(f"  Best Practices: {metrics.best_practices_compliance:.1f}/100")
    
    if project_health.security_issues:
        print(f"\nSecurity Issues Found:")
        for issue in project_health.security_issues:
            print(f"  ⚠️  {issue}")
    
    if project_health.performance_issues:
        print(f"\nPerformance Issues Found:")
        for issue in project_health.performance_issues:
            print(f"  🐌 {issue}")
    
    if project_health.recommendations:
        print(f"\nRecommendations:")
        for rec in project_health.recommendations:
            print(f"  💡 {rec}")
    
    # Cleanup
    import shutil
    shutil.rmtree(demo_project)


async def demo_checkpoint_system():
    """Demonstrate checkpoint and rollback system"""
    print("\n💾 === Checkpoint System Demo ===")
    
    # Create demo project
    demo_project = Path("checkpoint_demo")
    demo_project.mkdir(exist_ok=True)
    
    # Create initial file
    initial_content = "print('Initial version')"
    (demo_project / "script.py").write_text(initial_content)
    
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(demo_project)
    
    # Create initial checkpoint
    checkpoint1 = await checkpoint_manager.create_checkpoint("Initial working version", working_state=True)
    print(f"Created checkpoint: {checkpoint1.description}")
    
    # Modify file
    (demo_project / "script.py").write_text("print('Modified version with bug')\nundefined_variable")
    
    # Create checkpoint for broken state
    checkpoint2 = await checkpoint_manager.create_checkpoint("Broken version", working_state=False)
    print(f"Created checkpoint: {checkpoint2.description}")
    
    # Show current content
    current_content = (demo_project / "script.py").read_text()
    print(f"Current content: {repr(current_content)}")
    
    # Rollback to working state
    latest_working = checkpoint_manager.get_latest_working_checkpoint()
    if latest_working:
        print(f"Rolling back to: {latest_working.description}")
        success = await checkpoint_manager.rollback_to_checkpoint(latest_working.checkpoint_id)
        
        if success:
            restored_content = (demo_project / "script.py").read_text()
            print(f"Restored content: {repr(restored_content)}")
        else:
            print("Rollback failed!")
    
    # Show checkpoint history
    print(f"\nCheckpoint History:")
    for checkpoint in checkpoint_manager.rollback_manager.checkpoints:
        status = "✅ Working" if checkpoint.working_state else "❌ Broken"
        print(f"  {status} {checkpoint.description} ({checkpoint.timestamp.strftime('%H:%M:%S')})")
    
    # Cleanup
    import shutil
    shutil.rmtree(demo_project)


async def main():
    """Run all demonstrations"""
    print("🚀 Advanced Debugging System Demonstration")
    print("=" * 50)
    
    try:
        await demo_language_detection()
        await demo_error_analysis()
        await demo_parallel_testing()
        await demo_learning_system()
        await demo_quality_assessment()
        await demo_checkpoint_system()
        
        print("\n✅ All demonstrations completed successfully!")
        print("\nKey Improvements Made:")
        print("• 🔍 Advanced language/framework detection across 9+ languages")
        print("• 🔧 Intelligent error classification with AI-powered suggestions")
        print("• 🚀 Parallel execution of multiple testing strategies")
        print("• 🧠 Progressive learning system that improves over time")
        print("• 📊 Comprehensive project health and quality assessment")
        print("• 💾 Smart checkpoint/rollback system for safe experimentation")
        print("• 🎯 Language-agnostic debugging that works with any tech stack")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 