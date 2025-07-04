"""
Advanced Debugging and Testing System

This module provides comprehensive debugging capabilities including:
- Language and framework detection
- Intelligent error classification and analysis  
- Multi-strategy parallel testing
- Progressive learning from successful fixes
- Smart rollback and checkpoint management
- Comprehensive validation and quality assessment
"""

import os
import re
import json
import asyncio

import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Set
from datetime import datetime

from . import models
from . import config

from openai import OpenAI, AzureOpenAI

# Initialize OpenAI client
if os.getenv("AZURE_OPENAI_ENDPOINT"):
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT") or ""
    )
else:
    client = OpenAI(api_key=config.OPENAI_API_KEY, base_url="http://0.0.0.0:11437/v1")


class LanguageDetector:
    """Advanced language and framework detection system"""
    
    # Language patterns based on file extensions and content
    LANGUAGE_PATTERNS = {
        'python': {
            'extensions': ['.py', '.pyx', '.pyi'],
            'files': ['requirements.txt', 'setup.py', 'pyproject.toml', 'Pipfile'],
            'content_patterns': [r'import\s+\w+', r'from\s+\w+\s+import', r'def\s+\w+\(', r'class\s+\w+:']
        },
        'javascript': {
            'extensions': ['.js', '.mjs'],
            'files': ['package.json', 'package-lock.json', 'yarn.lock'],
            'content_patterns': [r'require\(', r'import\s+.*\s+from', r'export\s+', r'function\s+\w+\(']
        },
        'typescript': {
            'extensions': ['.ts', '.tsx'],
            'files': ['tsconfig.json'],
            'content_patterns': [r'interface\s+\w+', r'type\s+\w+\s*=', r'import\s+.*\s+from', r':\s*\w+\s*[=;]']
        },
        'go': {
            'extensions': ['.go'],
            'files': ['go.mod', 'go.sum'],
            'content_patterns': [r'package\s+\w+', r'import\s+\(', r'func\s+\w+\(', r'type\s+\w+\s+struct']
        },
        'rust': {
            'extensions': ['.rs'],
            'files': ['Cargo.toml', 'Cargo.lock'],
            'content_patterns': [r'fn\s+\w+\(', r'struct\s+\w+', r'impl\s+\w+', r'use\s+\w+::']
        },
        'java': {
            'extensions': ['.java'],
            'files': ['pom.xml', 'build.gradle'],
            'content_patterns': [r'public\s+class\s+\w+', r'import\s+\w+\.', r'public\s+static\s+void\s+main']
        },
        'csharp': {
            'extensions': ['.cs'],
            'files': ['*.csproj', '*.sln'],
            'content_patterns': [r'using\s+\w+;', r'public\s+class\s+\w+', r'namespace\s+\w+']
        },
        'php': {
            'extensions': ['.php'],
            'files': ['composer.json', 'composer.lock'],
            'content_patterns': [r'<\?php', r'function\s+\w+\(', r'class\s+\w+']
        },
        'ruby': {
            'extensions': ['.rb'],
            'files': ['Gemfile', 'Gemfile.lock'],
            'content_patterns': [r'def\s+\w+', r'class\s+\w+', r'require\s+']
        }
    }
    
    # Framework detection patterns
    FRAMEWORK_PATTERNS = {
        'fastapi': {
            'languages': ['python'],
            'indicators': ['fastapi', 'uvicorn', '@app.get', '@app.post', 'FastAPI()'],
            'files': ['main.py', 'app.py'],
            'category': 'web'
        },
        'django': {
            'languages': ['python'], 
            'indicators': ['django', 'manage.py', 'settings.py', 'from django.'],
            'files': ['manage.py', 'settings.py'],
            'category': 'web'
        },
        'flask': {
            'languages': ['python'],
            'indicators': ['flask', '@app.route', 'from flask import'],
            'files': ['app.py'],
            'category': 'web'
        },
        'express': {
            'languages': ['javascript', 'typescript'],
            'indicators': ['express', 'app.get(', 'app.post(', 'require("express")'],
            'files': ['server.js', 'app.js', 'index.js'],
            'category': 'web'
        },
        'react': {
            'languages': ['javascript', 'typescript'],
            'indicators': ['react', 'jsx', 'React.', 'import React'],
            'files': ['package.json'],
            'category': 'frontend'
        },
        'vue': {
            'languages': ['javascript', 'typescript'],
            'indicators': ['vue', '<template>', '<script>', 'Vue.'],
            'files': ['package.json'],
            'category': 'frontend'
        },
        'gin': {
            'languages': ['go'],
            'indicators': ['gin-gonic', 'gin.Default()', 'router.GET'],
            'files': ['main.go'],
            'category': 'web'
        },
        'actix': {
            'languages': ['rust'],
            'indicators': ['actix-web', 'HttpServer::new', 'App::new()'],
            'files': ['main.rs', 'src/main.rs'],
            'category': 'web'
        },
        'spring': {
            'languages': ['java'],
            'indicators': ['spring-boot', '@RestController', '@SpringBootApplication'],
            'files': ['pom.xml', 'build.gradle'],
            'category': 'web'
        }
    }

    @classmethod
    async def detect_tech_stack(cls, project_path: Path) -> models.TechStackDetection:
        """Detect the complete technology stack of a project"""
        
        # Get all project files
        project_files = {}
        file_extensions = set()
        
        for root, _, files in os.walk(project_path):
            if any(skip in root for skip in ['.git', '__pycache__', 'node_modules', '.venv', 'target', 'build']):
                continue
                
            for file in files:
                file_path = Path(root) / file
                relative_path = file_path.relative_to(project_path)
                
                try:
                    if file_path.stat().st_size < 1024 * 1024:  # Skip files larger than 1MB
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        project_files[str(relative_path)] = content
                        file_extensions.add(file_path.suffix.lower())
                except Exception:
                    continue

        # Detect languages
        detected_languages = []
        for lang, patterns in cls.LANGUAGE_PATTERNS.items():
            confidence = cls._calculate_language_confidence(lang, patterns, project_files, file_extensions)
            if confidence > 0.1:  # Minimum confidence threshold
                version = await cls._detect_language_version(lang, project_files)
                detected_languages.append(models.LanguageInfo(
                    language=lang,
                    version=version,
                    confidence=confidence
                ))
        
        # Sort by confidence and determine primary language
        detected_languages.sort(key=lambda x: x.confidence, reverse=True)
        primary_language = detected_languages[0] if detected_languages else models.LanguageInfo(
            language="unknown", confidence=0.0
        )
        secondary_languages = detected_languages[1:5]  # Top 4 secondary languages

        # Detect frameworks
        detected_frameworks = []
        for framework, patterns in cls.FRAMEWORK_PATTERNS.items():
            if primary_language.language in patterns['languages'] or any(
                lang.language in patterns['languages'] for lang in secondary_languages
            ):
                confidence = cls._calculate_framework_confidence(framework, patterns, project_files)
                if confidence > 0.2:  # Framework confidence threshold
                    version = await cls._detect_framework_version(framework, project_files)
                    detected_frameworks.append(models.FrameworkInfo(
                        name=framework,
                        version=version,
                        category=patterns['category'],
                        confidence=confidence
                    ))

        # Detect package managers, build tools, etc.
        package_managers = cls._detect_package_managers(project_files)
        build_tools = cls._detect_build_tools(project_files)
        deployment_tools = cls._detect_deployment_tools(project_files)
        databases = cls._detect_databases(project_files)

        return models.TechStackDetection(
            primary_language=primary_language,
            secondary_languages=secondary_languages,
            frameworks=detected_frameworks,
            package_managers=package_managers,
            build_tools=build_tools,
            deployment_tools=deployment_tools,
            databases=databases
        )

    @classmethod
    def _calculate_language_confidence(cls, language: str, patterns: Dict, 
                                     project_files: Dict[str, str], extensions: Set[str]) -> float:
        """Calculate confidence score for language detection"""
        score = 0.0
        
        # Extension matching (40% weight)
        ext_matches = sum(1 for ext in patterns['extensions'] if ext in extensions)
        if ext_matches > 0:
            score += 0.4 * min(ext_matches / len(patterns['extensions']), 1.0)
        
        # Special files (30% weight)
        file_matches = sum(1 for file in patterns['files'] if any(file in path for path in project_files.keys()))
        if file_matches > 0:
            score += 0.3 * min(file_matches / len(patterns['files']), 1.0)
        
        # Content pattern matching (30% weight)
        content_matches = 0
        total_content = '\n'.join(project_files.values())
        for pattern in patterns['content_patterns']:
            if re.search(pattern, total_content, re.MULTILINE):
                content_matches += 1
        
        if content_matches > 0:
            score += 0.3 * min(content_matches / len(patterns['content_patterns']), 1.0)
        
        return min(score, 1.0)

    @classmethod  
    def _calculate_framework_confidence(cls, framework: str, patterns: Dict, 
                                      project_files: Dict[str, str]) -> float:
        """Calculate confidence score for framework detection"""
        score = 0.0
        total_content = '\n'.join(project_files.values())
        
        # Check for framework indicators
        indicator_matches = sum(1 for indicator in patterns['indicators'] 
                              if indicator.lower() in total_content.lower())
        
        if indicator_matches > 0:
            score += 0.6 * min(indicator_matches / len(patterns['indicators']), 1.0)
        
        # Check for framework-specific files
        file_matches = sum(1 for file in patterns['files'] if any(file in path for path in project_files.keys()))
        if file_matches > 0:
            score += 0.4 * min(file_matches / len(patterns['files']), 1.0)
        
        return min(score, 1.0)

    @classmethod
    async def _detect_language_version(cls, language: str, project_files: Dict[str, str]) -> Optional[str]:
        """Detect language version from project files"""
        version_patterns = {
            'python': [
                (r'python_requires\s*=\s*["\']([^"\']+)["\']', 'setup.py'),
                (r'requires-python\s*=\s*["\']([^"\']+)["\']', 'pyproject.toml'),
                (r'"python":\s*"([^"]+)"', 'Pipfile')
            ],
            'node': [
                (r'"node":\s*"([^"]+)"', 'package.json'),
                (r'engine-strict.*"node":\s*"([^"]+)"', 'package.json')
            ],
            'go': [
                (r'go\s+(\d+\.\d+)', 'go.mod')
            ]
        }
        
        if language in version_patterns:
            for pattern, filename in version_patterns[language]:
                for path, content in project_files.items():
                    if filename in path:
                        match = re.search(pattern, content)
                        if match:
                            return match.group(1)
        
        return None

    @classmethod
    async def _detect_framework_version(cls, framework: str, project_files: Dict[str, str]) -> Optional[str]:
        """Detect framework version from project files"""
        # Check package.json for JS frameworks
        for path, content in project_files.items():
            if 'package.json' in path:
                try:
                    package_data = json.loads(content)
                    dependencies = {**package_data.get('dependencies', {}), 
                                  **package_data.get('devDependencies', {})}
                    if framework in dependencies:
                        return dependencies[framework].lstrip('^~>=')
                except json.JSONDecodeError:
                    pass
        
        # Check requirements.txt for Python frameworks  
        for path, content in project_files.items():
            if 'requirements.txt' in path:
                for line in content.split('\n'):
                    if framework in line.lower():
                        match = re.search(rf'{framework}[>=<]*([0-9.]+)', line, re.IGNORECASE)
                        if match:
                            return match.group(1)
        
        return None

    @classmethod
    def _detect_package_managers(cls, project_files: Dict[str, str]) -> List[str]:
        """Detect package managers used in the project"""
        managers = []
        indicators = {
            'npm': ['package.json', 'package-lock.json'],
            'yarn': ['yarn.lock'],
            'pip': ['requirements.txt', 'setup.py'],
            'pipenv': ['Pipfile'],
            'poetry': ['pyproject.toml'],
            'cargo': ['Cargo.toml'],
            'go': ['go.mod'],
            'maven': ['pom.xml'],
            'gradle': ['build.gradle'],
            'composer': ['composer.json']
        }
        
        for manager, files in indicators.items():
            if any(any(file in path for path in project_files.keys()) for file in files):
                managers.append(manager)
        
        return managers

    @classmethod
    def _detect_build_tools(cls, project_files: Dict[str, str]) -> List[str]:
        """Detect build tools used in the project"""
        tools = []
        indicators = {
            'webpack': ['webpack.config.js', 'webpack.config.ts'],
            'vite': ['vite.config.js', 'vite.config.ts'],
            'rollup': ['rollup.config.js'],
            'parcel': ['package.json'],  # Need to check content
            'esbuild': ['esbuild.config.js'],
            'make': ['Makefile'],
            'cmake': ['CMakeLists.txt'],
            'bazel': ['BUILD', 'WORKSPACE']
        }
        
        for tool, files in indicators.items():
            if any(any(file in path for path in project_files.keys()) for file in files):
                tools.append(tool)
        
        # Check package.json for additional build tools
        for path, content in project_files.items():
            if 'package.json' in path:
                try:
                    package_data = json.loads(content)
                    scripts = package_data.get('scripts', {})
                    dev_deps = package_data.get('devDependencies', {})
                    
                    for tool in ['parcel', 'rollup', 'esbuild']:
                        if tool in dev_deps or any(tool in script for script in scripts.values()):
                            if tool not in tools:
                                tools.append(tool)
                except json.JSONDecodeError:
                    pass
        
        return tools

    @classmethod
    def _detect_deployment_tools(cls, project_files: Dict[str, str]) -> List[str]:
        """Detect deployment tools and platforms"""
        tools = []
        indicators = {
            'docker': ['Dockerfile', 'docker-compose.yml'],
            'kubernetes': ['k8s/', 'kubernetes/', '*.yaml', '*.yml'],
            'terraform': ['*.tf', 'terraform/'],
            'ansible': ['playbook.yml', 'ansible/'],
            'vercel': ['vercel.json'],
            'netlify': ['netlify.toml', '_redirects'],
            'heroku': ['Procfile', 'app.json']
        }
        
        for tool, files in indicators.items():
            if any(any(pattern in path for path in project_files.keys()) for pattern in files):
                tools.append(tool)
        
        return tools

    @classmethod
    def _detect_databases(cls, project_files: Dict[str, str]) -> List[str]:
        """Detect database technologies"""
        databases = []
        content = '\n'.join(project_files.values()).lower()
        
        db_patterns = {
            'postgresql': ['postgresql', 'postgres', 'psycopg2', 'pg'],
            'mysql': ['mysql', 'pymysql', 'mysql2'],
            'sqlite': ['sqlite', 'sqlite3'],
            'mongodb': ['mongodb', 'mongoose', 'pymongo'],
            'redis': ['redis', 'redis-py'],
            'elasticsearch': ['elasticsearch', 'elastic'],
            'cassandra': ['cassandra'],
            'dynamodb': ['dynamodb', 'boto3']
        }
        
        for db, patterns in db_patterns.items():
            if any(pattern in content for pattern in patterns):
                databases.append(db)
        
        return databases


class ErrorAnalyzer:
    """Advanced error classification and analysis system"""
    
    ERROR_CATEGORIES = {
        'syntax': {
            'patterns': [
                r'SyntaxError', r'syntax error', r'unexpected token', r'invalid syntax',
                r'compilation error', r'parse error', r'missing semicolon'
            ],
            'severity': 'high'
        },
        'dependency': {
            'patterns': [
                r'ModuleNotFoundError', r'ImportError', r'cannot find module',
                r'package not found', r'dependency.*not.*found', r'missing dependency',
                r'version conflict', r'ENOENT.*node_modules'
            ],
            'severity': 'high'
        },
        'configuration': {
            'patterns': [
                r'configuration error', r'config.*not found', r'invalid configuration',
                r'missing environment variable', r'port.*already in use', r'connection refused'
            ],
            'severity': 'medium'
        },
        'runtime': {
            'patterns': [
                r'RuntimeError', r'NullPointerException', r'TypeError', r'ReferenceError',
                r'undefined.*function', r'null.*reference', r'segmentation fault'
            ],
            'severity': 'critical'
        },
        'permission': {
            'patterns': [
                r'permission denied', r'access denied', r'unauthorized',
                r'EACCES', r'EPERM'
            ],
            'severity': 'medium'
        },
        'network': {
            'patterns': [
                r'connection.*refused', r'timeout', r'network.*unreachable',
                r'DNS.*resolution', r'certificate.*error'
            ],
            'severity': 'medium'
        },
        'filesystem': {
            'patterns': [
                r'no such file', r'file not found', r'directory not found',
                r'ENOENT', r'disk.*full', r'permission denied'
            ],
            'severity': 'medium'
        }
    }

    @classmethod
    async def analyze_errors(cls, error_logs: str, tech_stack: models.TechStackDetection) -> List[models.ErrorAnalysis]:
        """Analyze error logs and classify errors"""
        
        # Split error logs into individual errors
        error_sections = cls._extract_error_sections(error_logs)
        
        analyses = []
        for error_text in error_sections:
            analysis = await cls._analyze_single_error(error_text, tech_stack)
            if analysis:
                analyses.append(analysis)
        
        return analyses

    @classmethod
    def _extract_error_sections(cls, error_logs: str) -> List[str]:
        """Extract individual error sections from logs"""
        # Common error delimiters
        delimiters = [
            r'Error:', r'ERROR:', r'FATAL:', r'FAIL:', r'Exception:',
            r'\[ERROR\]', r'\[FATAL\]', r'✗', r'❌'
        ]
        
        sections = []
        current_section = []
        
        for line in error_logs.split('\n'):
            line = line.strip()
            if not line:
                if current_section:
                    sections.append('\n'.join(current_section))
                    current_section = []
                continue
            
            # Check if line starts a new error
            is_new_error = any(re.search(delimiter, line, re.IGNORECASE) for delimiter in delimiters)
            
            if is_new_error and current_section:
                sections.append('\n'.join(current_section))
                current_section = [line]
            else:
                current_section.append(line)
        
        if current_section:
            sections.append('\n'.join(current_section))
        
        return [s for s in sections if len(s.strip()) > 10]  # Filter very short sections

    @classmethod
    async def _analyze_single_error(cls, error_text: str, tech_stack: models.TechStackDetection) -> Optional[models.ErrorAnalysis]:
        """Analyze a single error and classify it"""
        
        # Classify the error
        classification = cls._classify_error(error_text)
        if not classification:
            return None
        
        # Extract affected files
        affected_files = cls._extract_affected_files(error_text)
        
        # Generate AI-powered suggestions
        suggested_fixes = await cls._generate_ai_suggestions(error_text, classification, tech_stack)
        
        # Find related errors
        related_errors = cls._find_related_errors(error_text)
        
        # Determine root cause
        root_cause = await cls._determine_root_cause(error_text, classification, tech_stack)
        
        return models.ErrorAnalysis(
            original_error=error_text,
            classification=classification,
            affected_files=affected_files,
            suggested_fixes=suggested_fixes,
            related_errors=related_errors,
            root_cause=root_cause
        )

    @classmethod
    def _classify_error(cls, error_text: str) -> Optional[models.ErrorCategory]:
        """Classify an error into categories"""
        
        for category, info in cls.ERROR_CATEGORIES.items():
            matches = sum(1 for pattern in info['patterns'] 
                         if re.search(pattern, error_text, re.IGNORECASE))
            
            if matches > 0:
                confidence = min(matches / len(info['patterns']), 1.0) * 0.8 + 0.2
                
                # Determine subcategory based on specific patterns
                subcategory = cls._determine_subcategory(category, error_text)
                
                return models.ErrorCategory(
                    category=category,
                    subcategory=subcategory,
                    severity=info['severity'],
                    confidence=confidence
                )
        
        # Default classification for unrecognized errors
        return models.ErrorCategory(
            category="unknown",
            subcategory=None,
            severity="medium",
            confidence=0.1
        )

    @classmethod
    def _determine_subcategory(cls, category: str, error_text: str) -> Optional[str]:
        """Determine error subcategory based on specific patterns"""
        
        subcategory_patterns = {
            'dependency': {
                'missing_package': [r'cannot find module', r'ModuleNotFoundError', r'ImportError'],
                'version_conflict': [r'version.*conflict', r'incompatible.*version'],
                'installation': [r'failed to install', r'npm.*install.*failed']
            },
            'syntax': {
                'missing_bracket': [r'missing.*bracket', r'unclosed.*bracket'],
                'missing_semicolon': [r'missing.*semicolon', r'expected.*semicolon'],
                'invalid_character': [r'unexpected.*character', r'invalid.*character']
            },
            'configuration': {
                'env_variable': [r'environment.*variable', r'env.*not.*set'],
                'port_binding': [r'port.*already.*use', r'address.*already.*use'],
                'file_path': [r'config.*file.*not.*found', r'configuration.*missing']
            }
        }
        
        if category in subcategory_patterns:
            for subcategory, patterns in subcategory_patterns[category].items():
                if any(re.search(pattern, error_text, re.IGNORECASE) for pattern in patterns):
                    return subcategory
        
        return None

    @classmethod
    def _extract_affected_files(cls, error_text: str) -> List[str]:
        """Extract file paths mentioned in error text"""
        
        # Common file path patterns
        patterns = [
            r'File "([^"]+)"',  # Python style
            r'at ([^\s]+\.(?:js|ts|py|go|rs|java|cpp|h))',  # Various file extensions
            r'in ([^\s]+\.(?:js|ts|py|go|rs|java|cpp|h))',
            r'([^\s]+\.(?:js|ts|py|go|rs|java|cpp|h)):\d+',  # File with line number
            r'/([^/\s]+/)*[^/\s]+\.(?:js|ts|py|go|rs|java|cpp|h)',  # Full paths
        ]
        
        files = set()
        for pattern in patterns:
            matches = re.findall(pattern, error_text, re.IGNORECASE)
            files.update(matches)
        
        # Clean up file paths
        cleaned_files = []
        for file in files:
            if isinstance(file, tuple):
                file = file[-1]  # Get the last element if it's a tuple from regex groups
            
            # Remove common prefixes and clean path
            file = file.strip('/')
            if file and not file.startswith(('http://', 'https://')):
                cleaned_files.append(file)
        
        return list(set(cleaned_files))

    @classmethod
    async def _generate_ai_suggestions(cls, error_text: str, classification: models.ErrorCategory, 
                                     tech_stack: models.TechStackDetection) -> List[str]:
        """Generate AI-powered fix suggestions"""
        
        prompt = f"""
        You are an expert debugging assistant. Analyze this error and provide specific, actionable fix suggestions.

        Error Classification: {classification.category} ({classification.subcategory or 'general'})
        Severity: {classification.severity}
        
        Technology Stack:
        - Primary Language: {tech_stack.primary_language.language}
        - Frameworks: {[f.name for f in tech_stack.frameworks]}
        - Package Managers: {tech_stack.package_managers}
        
        Error Details:
        {error_text}
        
        Provide 3-5 specific, actionable suggestions to fix this error. Focus on:
        1. The most likely root cause
        2. Specific commands or code changes needed
        3. Alternative approaches if the primary fix doesn't work
        
        Return your response as a JSON array of strings, where each string is a specific fix suggestion.
        """
        
        try:
            completion = client.chat.completions.create(
                model=config.DEFAULT_MODEL,
                messages=[
                    {"role": "system", "content": "You are a debugging expert. Provide specific, actionable fix suggestions as JSON array."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            response = completion.choices[0].message.content
            if response:
                suggestions = json.loads(response)
                return suggestions if isinstance(suggestions, list) else [response]
        except Exception as e:
            print(f"Error generating AI suggestions: {e}")
        
        # Fallback suggestions based on classification
        return cls._get_fallback_suggestions(classification)

    @classmethod
    def _get_fallback_suggestions(cls, classification: models.ErrorCategory) -> List[str]:
        """Provide fallback suggestions when AI generation fails"""
        
        fallback_suggestions = {
            'dependency': [
                "Check if all required dependencies are installed",
                "Verify package versions in package.json/requirements.txt",
                "Clear package cache and reinstall dependencies",
                "Check for conflicting package versions"
            ],
            'syntax': [
                "Review code for syntax errors (missing brackets, semicolons)",
                "Check indentation and code formatting",
                "Validate file encoding (UTF-8)",
                "Use a linter to identify syntax issues"
            ],
            'configuration': [
                "Verify all required environment variables are set",
                "Check configuration file paths and permissions",
                "Validate configuration file syntax",
                "Ensure all required services are running"
            ],
            'runtime': [
                "Add error handling and null checks",
                "Verify variable initialization",
                "Check function/method signatures",
                "Review variable scoping"
            ]
        }
        
        return fallback_suggestions.get(classification.category, [
            "Review error logs for more specific information",
            "Check documentation for similar issues",
            "Verify system requirements and dependencies",
            "Consider seeking help from community forums"
        ])

    @classmethod
    def _find_related_errors(cls, error_text: str) -> List[str]:
        """Find patterns that commonly occur with this error"""
        
        # This could be enhanced with a machine learning model
        # For now, use simple pattern matching
        related_patterns = {
            'ModuleNotFoundError': ['ImportError', 'package not found'],
            'SyntaxError': ['IndentationError', 'invalid syntax'],
            'connection refused': ['port already in use', 'service not running'],
            'permission denied': ['access denied', 'insufficient privileges']
        }
        
        related = []
        error_lower = error_text.lower()
        
        for primary, related_list in related_patterns.items():
            if primary.lower() in error_lower:
                related.extend(related_list)
        
        return related

    @classmethod
    async def _determine_root_cause(cls, error_text: str, classification: models.ErrorCategory,
                                   tech_stack: models.TechStackDetection) -> Optional[str]:
        """Determine the most likely root cause of the error"""
        
        prompt = f"""
        Analyze this error and determine the most likely root cause.

        Error Category: {classification.category}
        Technology: {tech_stack.primary_language.language}
        
        Error:
        {error_text}
        
        Provide a single, concise sentence describing the root cause.
        """
        
        try:
            completion = client.chat.completions.create(
                model=config.DEFAULT_MODEL,
                messages=[
                    {"role": "system", "content": "You are a debugging expert. Identify root causes concisely."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            response = completion.choices[0].message.content
            return response.strip() if response else None
        except Exception as e:
            print(f"Error determining root cause: {e}")
            return None


# Continue with more classes in the next part...


class TestStrategyManager:
    """Manages and coordinates different testing strategies"""
    
    @classmethod
    def get_language_strategies(cls, tech_stack: models.TechStackDetection) -> List[models.TestStrategy]:
        """Get applicable testing strategies based on technology stack"""
        
        strategies = []
        language = tech_stack.primary_language.language
        frameworks = [f.name for f in tech_stack.frameworks]
        
        # Base strategies for all languages
        base_strategies = [
            models.TestStrategy(
                strategy_id="build_validation",
                name="Build Validation",
                description="Validate that the project builds successfully",
                applicable_languages=["all"],
                applicable_frameworks=["all"],
                priority=1,
                estimated_time=60,
                success_rate=0.9
            ),
            models.TestStrategy(
                strategy_id="syntax_check",
                name="Syntax Check",
                description="Check for syntax errors and basic compilation issues",
                applicable_languages=["all"],
                applicable_frameworks=["all"],
                priority=2,
                estimated_time=30,
                success_rate=0.85
            )
        ]
        
        # Language-specific strategies
        language_strategies = {
            'python': [
                models.TestStrategy(
                    strategy_id="python_lint",
                    name="Python Linting",
                    description="Run pylint/flake8 for code quality checks",
                    applicable_languages=["python"],
                    applicable_frameworks=["all"],
                    priority=3,
                    estimated_time=45,
                    success_rate=0.75
                ),
                models.TestStrategy(
                    strategy_id="python_deps",
                    name="Python Dependencies",
                    description="Check and install Python dependencies",
                    applicable_languages=["python"],
                    applicable_frameworks=["all"],
                    priority=2,
                    estimated_time=90,
                    success_rate=0.8
                ),
                models.TestStrategy(
                    strategy_id="python_imports",
                    name="Import Validation",
                    description="Validate all import statements",
                    applicable_languages=["python"],
                    applicable_frameworks=["all"],
                    priority=4,
                    estimated_time=20,
                    success_rate=0.9
                )
            ],
            'javascript': [
                models.TestStrategy(
                    strategy_id="npm_install",
                    name="NPM Install",
                    description="Install Node.js dependencies",
                    applicable_languages=["javascript", "typescript"],
                    applicable_frameworks=["all"],
                    priority=2,
                    estimated_time=120,
                    success_rate=0.85
                ),
                models.TestStrategy(
                    strategy_id="eslint_check",
                    name="ESLint Check",
                    description="Run ESLint for code quality",
                    applicable_languages=["javascript", "typescript"],
                    applicable_frameworks=["all"],
                    priority=3,
                    estimated_time=30,
                    success_rate=0.7
                ),
                models.TestStrategy(
                    strategy_id="typescript_compile",
                    name="TypeScript Compilation",
                    description="Compile TypeScript to JavaScript",
                    applicable_languages=["typescript"],
                    applicable_frameworks=["all"],
                    priority=2,
                    estimated_time=60,
                    success_rate=0.8
                )
            ],
            'go': [
                models.TestStrategy(
                    strategy_id="go_mod_download",
                    name="Go Mod Download",
                    description="Download Go module dependencies",
                    applicable_languages=["go"],
                    applicable_frameworks=["all"],
                    priority=2,
                    estimated_time=60,
                    success_rate=0.9
                ),
                models.TestStrategy(
                    strategy_id="go_build",
                    name="Go Build",
                    description="Build Go application",
                    applicable_languages=["go"],
                    applicable_frameworks=["all"],
                    priority=1,
                    estimated_time=45,
                    success_rate=0.85
                ),
                models.TestStrategy(
                    strategy_id="go_vet",
                    name="Go Vet",
                    description="Run go vet for suspicious constructs",
                    applicable_languages=["go"],
                    applicable_frameworks=["all"],
                    priority=3,
                    estimated_time=30,
                    success_rate=0.8
                )
            ],
            'rust': [
                models.TestStrategy(
                    strategy_id="cargo_check",
                    name="Cargo Check",
                    description="Check Rust code without building",
                    applicable_languages=["rust"],
                    applicable_frameworks=["all"],
                    priority=2,
                    estimated_time=45,
                    success_rate=0.85
                ),
                models.TestStrategy(
                    strategy_id="cargo_build",
                    name="Cargo Build",
                    description="Build Rust application",
                    applicable_languages=["rust"],
                    applicable_frameworks=["all"],
                    priority=1,
                    estimated_time=90,
                    success_rate=0.8
                ),
                models.TestStrategy(
                    strategy_id="clippy_check",
                    name="Clippy Check",
                    description="Run Clippy for Rust best practices",
                    applicable_languages=["rust"],
                    applicable_frameworks=["all"],
                    priority=3,
                    estimated_time=60,
                    success_rate=0.75
                )
            ]
        }
        
        # Framework-specific strategies
        framework_strategies = {
            'fastapi': [
                models.TestStrategy(
                    strategy_id="fastapi_startup",
                    name="FastAPI Startup Test",
                    description="Test FastAPI application startup",
                    applicable_languages=["python"],
                    applicable_frameworks=["fastapi"],
                    priority=2,
                    estimated_time=30,
                    success_rate=0.8
                )
            ],
            'django': [
                models.TestStrategy(
                    strategy_id="django_check",
                    name="Django Check",
                    description="Run Django system checks",
                    applicable_languages=["python"],
                    applicable_frameworks=["django"],
                    priority=2,
                    estimated_time=45,
                    success_rate=0.85
                )
            ],
            'express': [
                models.TestStrategy(
                    strategy_id="express_startup",
                    name="Express Startup Test",
                    description="Test Express server startup",
                    applicable_languages=["javascript", "typescript"],
                    applicable_frameworks=["express"],
                    priority=2,
                    estimated_time=30,
                    success_rate=0.8
                )
            ]
        }
        
        # Add applicable strategies
        strategies.extend(base_strategies)
        
        if language in language_strategies:
            strategies.extend(language_strategies[language])
        
        for framework in frameworks:
            if framework in framework_strategies:
                strategies.extend(framework_strategies[framework])
        
        # Sort by priority and success rate
        strategies.sort(key=lambda x: (x.priority, -x.success_rate))
        
        return strategies


class ParallelTestExecutor:
    """Executes multiple testing strategies in parallel"""
    
    @classmethod
    async def execute_parallel_strategies(cls, project_path: Path, execution_plan: models.ParallelTestExecution,
                                        tech_stack: models.TechStackDetection) -> List[models.TestResult]:
        """Execute multiple testing strategies in parallel"""
        
        results = []
        semaphore = asyncio.Semaphore(execution_plan.max_parallel)
        
        async def execute_single_strategy(strategy: models.TestStrategy) -> models.TestResult:
            async with semaphore:
                return await cls._execute_strategy(project_path, strategy, tech_stack)
        
        # Create tasks for all strategies
        tasks = [execute_single_strategy(strategy) for strategy in execution_plan.strategies]
        
        try:
            # Wait for all tasks with timeout
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=execution_plan.timeout
            )
        except asyncio.TimeoutError:
            print(f"Parallel execution timed out after {execution_plan.timeout} seconds")
            results = [models.TestResult(
                strategy_id="timeout",
                success=False,
                execution_time=execution_plan.timeout,
                output="",
                error_message="Execution timed out",
                confidence_score=0.0
            )]
        
        # Filter out exceptions and return valid results
        valid_results = [r for r in results if isinstance(r, models.TestResult)]
        return valid_results

    @classmethod
    async def _execute_strategy(cls, project_path: Path, strategy: models.TestStrategy,
                              tech_stack: models.TechStackDetection) -> models.TestResult:
        """Execute a single testing strategy"""
        
        start_time = datetime.utcnow()
        
        try:
            commands = cls._get_strategy_commands(strategy, tech_stack)
            success, output, files_modified = await cls._run_strategy_commands(project_path, commands)
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            confidence_score = cls._calculate_confidence_score(success, strategy, output)
            
            return models.TestResult(
                strategy_id=strategy.strategy_id,
                success=success,
                execution_time=execution_time,
                output=output,
                error_message=None if success else output,
                files_modified=files_modified,
                confidence_score=confidence_score
            )
            
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            return models.TestResult(
                strategy_id=strategy.strategy_id,
                success=False,
                execution_time=execution_time,
                output="",
                error_message=str(e),
                confidence_score=0.0
            )

    @classmethod
    def _get_strategy_commands(cls, strategy: models.TestStrategy, 
                             tech_stack: models.TechStackDetection) -> List[str]:
        """Get commands to execute for a specific strategy"""
        
        commands_map = {
            'build_validation': cls._get_build_commands(tech_stack),
            'syntax_check': cls._get_syntax_check_commands(tech_stack),
            'python_lint': ['python -m flake8 .', 'python -m pylint **/*.py'],
            'python_deps': ['pip install -r requirements.txt', 'pip check'],
            'python_imports': ['python -c "import ast; [compile(open(f).read(), f, \"exec\") for f in glob.glob(\"**/*.py\", recursive=True)]"'],
            'npm_install': ['npm install', 'npm audit fix'],
            'eslint_check': ['npx eslint .'],
            'typescript_compile': ['npx tsc --noEmit'],
            'go_mod_download': ['go mod download', 'go mod verify'],
            'go_build': ['go build ./...'],
            'go_vet': ['go vet ./...'],
            'cargo_check': ['cargo check'],
            'cargo_build': ['cargo build'],
            'clippy_check': ['cargo clippy'],
            'fastapi_startup': ['python -c "from main import app; print(\"FastAPI app loaded successfully\")"'],
            'django_check': ['python manage.py check'],
            'express_startup': ['node -e "const app = require(\"./app\"); console.log(\"Express app loaded\")"']
        }
        
        return commands_map.get(strategy.strategy_id, ['echo "No commands defined for strategy"'])

    @classmethod
    def _get_build_commands(cls, tech_stack: models.TechStackDetection) -> List[str]:
        """Get build commands based on technology stack"""
        
        language = tech_stack.primary_language.language
        
        build_commands = {
            'python': ['python -m py_compile **/*.py'],
            'javascript': ['npm run build'] if 'npm' in tech_stack.package_managers else ['node --check **/*.js'],
            'typescript': ['npx tsc'],
            'go': ['go build ./...'],
            'rust': ['cargo build'],
            'java': ['javac **/*.java'],
            'csharp': ['dotnet build']
        }
        
        return build_commands.get(language, ['echo "Build not configured for this language"'])

    @classmethod
    def _get_syntax_check_commands(cls, tech_stack: models.TechStackDetection) -> List[str]:
        """Get syntax check commands based on technology stack"""
        
        language = tech_stack.primary_language.language
        
        syntax_commands = {
            'python': ['python -m py_compile **/*.py'],
            'javascript': ['node --check **/*.js'],
            'typescript': ['npx tsc --noEmit'],
            'go': ['go fmt -d .'],
            'rust': ['cargo check --quiet'],
            'java': ['javac -Xstdout /dev/null **/*.java'],
            'csharp': ['dotnet build --verbosity quiet']
        }
        
        return syntax_commands.get(language, ['echo "Syntax check not configured"'])

    @classmethod
    async def _run_strategy_commands(cls, project_path: Path, commands: List[str]) -> Tuple[bool, str, List[str]]:
        """Run strategy commands and return results"""
        
        output_lines = []
        files_modified = []
        overall_success = True
        
        for command in commands:
            try:
                # Track files before command
                files_before = set(cls._get_all_files(project_path))
                
                process = await asyncio.create_subprocess_shell(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=str(project_path)
                )
                
                stdout, stderr = await process.communicate()
                stdout_str = stdout.decode('utf-8', errors='ignore')
                stderr_str = stderr.decode('utf-8', errors='ignore')
                
                output_lines.append(f"Command: {command}")
                output_lines.append(f"Exit code: {process.returncode}")
                if stdout_str:
                    output_lines.append(f"STDOUT:\n{stdout_str}")
                if stderr_str:
                    output_lines.append(f"STDERR:\n{stderr_str}")
                
                if process.returncode != 0:
                    overall_success = False
                
                # Track files after command
                files_after = set(cls._get_all_files(project_path))
                modified = list(files_after - files_before)
                files_modified.extend(modified)
                
            except Exception as e:
                output_lines.append(f"Error executing command '{command}': {str(e)}")
                overall_success = False
        
        return overall_success, '\n'.join(output_lines), files_modified

    @classmethod
    def _get_all_files(cls, project_path: Path) -> List[str]:
        """Get list of all files in project"""
        files = []
        for root, _, filenames in os.walk(project_path):
            for filename in filenames:
                file_path = Path(root) / filename
                relative_path = file_path.relative_to(project_path)
                files.append(str(relative_path))
        return files

    @classmethod
    def _calculate_confidence_score(cls, success: bool, strategy: models.TestStrategy, output: str) -> float:
        """Calculate confidence score for test result"""
        
        if not success:
            return 0.0
        
        # Base score from strategy success rate
        base_score = strategy.success_rate
        
        # Adjust based on output quality
        output_lower = output.lower()
        positive_indicators = ['success', 'passed', 'ok', 'completed', 'built']
        negative_indicators = ['warning', 'deprecated', 'unstable']
        
        positive_count = sum(1 for indicator in positive_indicators if indicator in output_lower)
        negative_count = sum(1 for indicator in negative_indicators if indicator in output_lower)
        
        # Adjust score
        score_adjustment = (positive_count * 0.1) - (negative_count * 0.05)
        final_score = min(max(base_score + score_adjustment, 0.0), 1.0)
        
        return final_score


class LearningSystem:
    """Progressive learning system that improves debugging over time"""
    
    def __init__(self):
        self.patterns_db_path = Path("learning_patterns.json")
        self.patterns: Dict[str, models.FixPattern] = {}
        self.load_patterns()
    
    def load_patterns(self):
        """Load learning patterns from persistent storage"""
        if self.patterns_db_path.exists():
            try:
                with open(self.patterns_db_path, 'r') as f:
                    patterns_data = json.load(f)
                    
                for pattern_data in patterns_data:
                    pattern = models.FixPattern.model_validate(pattern_data)
                    self.patterns[pattern.pattern_id] = pattern
                    
            except Exception as e:
                print(f"Error loading learning patterns: {e}")
    
    def save_patterns(self):
        """Save learning patterns to persistent storage"""
        try:
            patterns_data = [pattern.model_dump() for pattern in self.patterns.values()]
            with open(self.patterns_db_path, 'w') as f:
                json.dump(patterns_data, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving learning patterns: {e}")
    
    def find_applicable_patterns(self, error_analysis: models.ErrorAnalysis, 
                               tech_stack: models.TechStackDetection) -> List[models.FixPattern]:
        """Find patterns applicable to the current error"""
        
        applicable_patterns = []
        error_signature = self._create_error_signature(error_analysis)
        
        for pattern in self.patterns.values():
            # Check language compatibility
            if pattern.language != tech_stack.primary_language.language:
                continue
            
            # Check framework compatibility if specified
            if pattern.framework:
                framework_names = [f.name for f in tech_stack.frameworks]
                if pattern.framework not in framework_names:
                    continue
            
            # Check error signature similarity
            similarity = self._calculate_signature_similarity(error_signature, pattern.error_signature)
            if similarity > 0.7:  # Similarity threshold
                applicable_patterns.append(pattern)
        
        # Sort by success rate and recency
        applicable_patterns.sort(
            key=lambda p: (p.success_count / max(p.success_count + p.failure_count, 1), p.last_used),
            reverse=True
        )
        
        return applicable_patterns[:5]  # Return top 5 patterns
    
    def learn_from_success(self, error_analysis: models.ErrorAnalysis, 
                          successful_changes: List[Dict[str, Any]],
                          tech_stack: models.TechStackDetection) -> models.FixPattern:
        """Learn from a successful fix"""
        
        error_signature = self._create_error_signature(error_analysis)
        
        # Check if we already have a similar pattern
        existing_pattern = None
        for pattern in self.patterns.values():
            if (pattern.language == tech_stack.primary_language.language and 
                self._calculate_signature_similarity(error_signature, pattern.error_signature) > 0.8):
                existing_pattern = pattern
                break
        
        if existing_pattern:
            # Update existing pattern
            existing_pattern.success_count += 1
            existing_pattern.last_used = datetime.utcnow()
            existing_pattern.code_changes.extend(successful_changes)
        else:
            # Create new pattern
            framework = tech_stack.frameworks[0].name if tech_stack.frameworks else None
            new_pattern = models.FixPattern(
                error_signature=error_signature,
                language=tech_stack.primary_language.language,
                framework=framework,
                fix_description=f"Fix for {error_analysis.classification.category} error",
                code_changes=successful_changes
            )
            self.patterns[new_pattern.pattern_id] = new_pattern
            existing_pattern = new_pattern
        
        self.save_patterns()
        return existing_pattern
    
    def learn_from_failure(self, pattern_id: str):
        """Learn from a failed pattern application"""
        if pattern_id in self.patterns:
            self.patterns[pattern_id].failure_count += 1
            self.patterns[pattern_id].last_used = datetime.utcnow()
            self.save_patterns()
    
    def _create_error_signature(self, error_analysis: models.ErrorAnalysis) -> str:
        """Create a signature for an error that can be used for matching"""
        
        # Normalize error message
        error_msg = error_analysis.original_error.lower()
        
        # Remove variable parts (numbers, paths, specific names)
        normalized = re.sub(r'\d+', 'NUMBER', error_msg)
        normalized = re.sub(r'/[^\s]+', 'PATH', normalized)
        normalized = re.sub(r'[a-zA-Z_]\w*\.[a-zA-Z_]\w*', 'MODULE.ATTR', normalized)
        
        # Create signature with classification
        signature_parts = [
            error_analysis.classification.category,
            error_analysis.classification.subcategory or "",
            normalized[:200]  # Truncate to avoid overly long signatures
        ]
        
        return " | ".join(signature_parts)
    
    def _calculate_signature_similarity(self, sig1: str, sig2: str) -> float:
        """Calculate similarity between two error signatures"""
        
        # Simple word-based similarity
        words1 = set(sig1.lower().split())
        words2 = set(sig2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)


class CheckpointManager:
    """Manages project checkpoints for smart rollback"""
    
    def __init__(self, project_path: Path):
        self.rollback_manager = models.RollbackManager(
            project_path=str(project_path),
            checkpoints=[],
            max_checkpoints=10
        )
        self.project_path = project_path
    
    async def create_checkpoint(self, description: str, working_state: bool = False) -> models.ProjectCheckpoint:
        """Create a checkpoint of the current project state"""
        
        # Read all project files
        files_snapshot = {}
        for root, _, files in os.walk(self.project_path):
            if any(skip in root for skip in ['.git', '__pycache__', 'node_modules', '.venv']):
                continue
                
            for file in files:
                file_path = Path(root) / file
                relative_path = file_path.relative_to(self.project_path)
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    files_snapshot[str(relative_path)] = content
                except Exception as e:
                    print(f"Warning: Could not read file {file_path}: {e}")
        
        checkpoint = models.ProjectCheckpoint(
            description=description,
            files_snapshot=files_snapshot,
            working_state=working_state
        )
        
        # Add checkpoint and manage size
        self.rollback_manager.checkpoints.append(checkpoint)
        if len(self.rollback_manager.checkpoints) > self.rollback_manager.max_checkpoints:
            self.rollback_manager.checkpoints.pop(0)  # Remove oldest
        
        self.rollback_manager.current_checkpoint = checkpoint.checkpoint_id
        
        return checkpoint
    
    async def rollback_to_checkpoint(self, checkpoint_id: str) -> bool:
        """Rollback project to a specific checkpoint"""
        
        target_checkpoint = None
        for checkpoint in self.rollback_manager.checkpoints:
            if checkpoint.checkpoint_id == checkpoint_id:
                target_checkpoint = checkpoint
                break
        
        if not target_checkpoint:
            print(f"Checkpoint {checkpoint_id} not found")
            return False
        
        try:
            # Remove all current files
            for root, dirs, files in os.walk(self.project_path):
                if any(skip in root for skip in ['.git']):
                    continue
                    
                for file in files:
                    file_path = Path(root) / file
                    file_path.unlink()
            
            # Restore files from checkpoint
            for relative_path, content in target_checkpoint.files_snapshot.items():
                file_path = self.project_path / relative_path
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            
            self.rollback_manager.current_checkpoint = checkpoint_id
            print(f"Successfully rolled back to checkpoint: {target_checkpoint.description}")
            return True
            
        except Exception as e:
            print(f"Error during rollback: {e}")
            return False
    
    def get_working_checkpoints(self) -> List[models.ProjectCheckpoint]:
        """Get all checkpoints marked as working states"""
        return [cp for cp in self.rollback_manager.checkpoints if cp.working_state]
    
    def get_latest_working_checkpoint(self) -> Optional[models.ProjectCheckpoint]:
        """Get the most recent working checkpoint"""
        working_checkpoints = self.get_working_checkpoints()
        if working_checkpoints:
            return max(working_checkpoints, key=lambda cp: cp.timestamp)
        return None


class QualityAssessor:
    """Assesses code quality and project health"""
    
    @classmethod
    async def assess_project_health(cls, project_path: Path, 
                                  tech_stack: models.TechStackDetection) -> models.ProjectHealth:
        """Perform comprehensive project health assessment"""
        
        # Run quality metrics assessment
        quality_metrics = await cls._assess_code_quality(project_path, tech_stack)
        
        # Test build success
        build_success = await cls._test_build_success(project_path, tech_stack)
        
        # Test if tests are passing (if tests exist)
        tests_passing = await cls._test_tests_passing(project_path, tech_stack)
        
        # Security assessment
        security_issues = await cls._assess_security(project_path, tech_stack)
        
        # Performance assessment
        performance_issues = await cls._assess_performance(project_path, tech_stack)
        
        # Generate recommendations
        recommendations = await cls._generate_recommendations(
            quality_metrics, build_success, tests_passing, security_issues, performance_issues
        )
        
        # Calculate overall health score
        overall_health = cls._calculate_overall_health(
            quality_metrics, build_success, tests_passing, security_issues, performance_issues
        )
        
        return models.ProjectHealth(
            overall_health=overall_health,
            quality_metrics=quality_metrics,
            build_success=build_success,
            tests_passing=tests_passing,
            security_issues=security_issues,
            performance_issues=performance_issues,
            recommendations=recommendations
        )
    
    @classmethod
    async def _assess_code_quality(cls, project_path: Path, 
                                 tech_stack: models.TechStackDetection) -> models.CodeQualityMetrics:
        """Assess code quality metrics"""
        
        # This would integrate with various code quality tools
        # For now, providing estimated metrics based on project structure
        
        language = tech_stack.primary_language.language
        
        # Count files and lines of code
        total_files = 0
        total_lines = 0
        
        for root, _, files in os.walk(project_path):
            if any(skip in root for skip in ['.git', '__pycache__', 'node_modules', '.venv']):
                continue
                
            for file in files:
                if file.endswith(('.py', '.js', '.ts', '.go', '.rs', '.java', '.cs')):
                    total_files += 1
                    try:
                        with open(Path(root) / file, 'r', encoding='utf-8', errors='ignore') as f:
                            total_lines += len(f.readlines())
                    except Exception:
                        pass
        
        # Estimate metrics based on project size and structure
        complexity_score = max(100 - (total_lines / 100), 0)  # Simpler = higher score
        maintainability_score = min(total_files * 10, 100)  # More files = potentially better organized
        
        # Mock other metrics
        test_coverage = 0.0  # Would need actual test analysis
        security_score = 85.0  # Default good score
        performance_score = 80.0  # Default good score
        best_practices_compliance = 75.0  # Default score
        
        return models.CodeQualityMetrics(
            complexity_score=complexity_score,
            maintainability_score=maintainability_score,
            test_coverage=test_coverage,
            security_score=security_score,
            performance_score=performance_score,
            best_practices_compliance=best_practices_compliance
        )
    
    @classmethod
    async def _test_build_success(cls, project_path: Path, 
                                tech_stack: models.TechStackDetection) -> bool:
        """Test if the project builds successfully"""
        
        language = tech_stack.primary_language.language
        
        build_commands = {
            'python': 'python -m py_compile **/*.py',
            'javascript': 'npm run build' if 'npm' in tech_stack.package_managers else 'node --check **/*.js',
            'typescript': 'npx tsc --noEmit',
            'go': 'go build ./...',
            'rust': 'cargo check',
            'java': 'javac **/*.java'
        }
        
        command = build_commands.get(language)
        if not command:
            return True  # Assume success if no build command
        
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(project_path)
            )
            
            await process.communicate()
            return process.returncode == 0
            
        except Exception:
            return False
    
    @classmethod
    async def _test_tests_passing(cls, project_path: Path, 
                                tech_stack: models.TechStackDetection) -> bool:
        """Test if project tests are passing"""
        
        language = tech_stack.primary_language.language
        
        test_commands = {
            'python': 'python -m pytest',
            'javascript': 'npm test',
            'typescript': 'npm test',
            'go': 'go test ./...',
            'rust': 'cargo test',
            'java': 'mvn test'
        }
        
        # Check if test files exist
        test_indicators = {
            'python': ['test_*.py', '*_test.py', 'tests/'],
            'javascript': ['*.test.js', '*.spec.js', 'test/', 'tests/'],
            'typescript': ['*.test.ts', '*.spec.ts', 'test/', 'tests/'],
            'go': ['*_test.go'],
            'rust': ['tests/'],
            'java': ['src/test/']
        }
        
        has_tests = False
        if language in test_indicators:
            for pattern in test_indicators[language]:
                if list(project_path.rglob(pattern)):
                    has_tests = True
                    break
        
        if not has_tests:
            return True  # No tests to fail
        
        command = test_commands.get(language)
        if not command:
            return True
        
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(project_path)
            )
            
            await process.communicate()
            return process.returncode == 0
            
        except Exception:
            return False
    
    @classmethod
    async def _assess_security(cls, project_path: Path, 
                             tech_stack: models.TechStackDetection) -> List[str]:
        """Assess security issues"""
        
        issues = []
        
        # Check for common security issues
        security_patterns = {
            'hardcoded_secrets': [
                r'password\s*=\s*["\'][^"\']{8,}["\']',
                r'api_key\s*=\s*["\'][^"\']{20,}["\']',
                r'secret\s*=\s*["\'][^"\']{16,}["\']'
            ],
            'insecure_http': [
                r'http://[^/\s]+',
                r'urllib.*http://'
            ]
        }
        
        # Read all source files and check for patterns
        for root, _, files in os.walk(project_path):
            if any(skip in root for skip in ['.git', '__pycache__', 'node_modules', '.venv']):
                continue
                
            for file in files:
                if file.endswith(('.py', '.js', '.ts', '.go', '.rs', '.java', '.cs')):
                    try:
                        with open(Path(root) / file, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            
                        for issue_type, patterns in security_patterns.items():
                            for pattern in patterns:
                                if re.search(pattern, content, re.IGNORECASE):
                                    issues.append(f"{issue_type} detected in {file}")
                                    break
                    except Exception:
                        pass
        
        return issues
    
    @classmethod
    async def _assess_performance(cls, project_path: Path, 
                                tech_stack: models.TechStackDetection) -> List[str]:
        """Assess performance issues"""
        
        issues = []
        
        # Check for common performance issues
        performance_patterns = {
            'python': [
                (r'for.*in.*range\(len\(', 'Use enumerate() instead of range(len())'),
                (r'\.append\(.*\)\s*$', 'Consider using list comprehension'),
            ],
            'javascript': [
                (r'for\s*\(\s*var\s+\w+\s*=\s*0', 'Consider using for...of or forEach'),
                (r'document\.getElementById', 'Consider caching DOM queries'),
            ]
        }
        
        language = tech_stack.primary_language.language
        if language not in performance_patterns:
            return issues
        
        patterns = performance_patterns[language]
        
        # Check source files for performance anti-patterns
        for root, _, files in os.walk(project_path):
            if any(skip in root for skip in ['.git', '__pycache__', 'node_modules', '.venv']):
                continue
                
            for file in files:
                file_ext = Path(file).suffix
                language_extensions = {
                    'python': ['.py'],
                    'javascript': ['.js'],
                    'typescript': ['.ts']
                }
                
                if file_ext not in language_extensions.get(language, []):
                    continue
                
                try:
                    with open(Path(root) / file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        
                    for pattern, description in patterns:
                        if re.search(pattern, content, re.MULTILINE):
                            issues.append(f"{description} in {file}")
                            
                except Exception:
                    pass
        
        return issues
    
    @classmethod
    async def _generate_recommendations(cls, quality_metrics: models.CodeQualityMetrics,
                                      build_success: bool, tests_passing: bool,
                                      security_issues: List[str], performance_issues: List[str]) -> List[str]:
        """Generate improvement recommendations"""
        
        recommendations = []
        
        if not build_success:
            recommendations.append("Fix build errors to ensure project compiles successfully")
        
        if not tests_passing:
            recommendations.append("Fix failing tests to ensure code reliability")
        
        if quality_metrics.complexity_score < 50:
            recommendations.append("Refactor complex code to improve maintainability")
        
        if quality_metrics.test_coverage < 80:
            recommendations.append("Increase test coverage to improve code reliability")
        
        if security_issues:
            recommendations.append("Address security issues found in code analysis")
        
        if performance_issues:
            recommendations.append("Optimize performance by addressing identified bottlenecks")
        
        if quality_metrics.best_practices_compliance < 80:
            recommendations.append("Improve adherence to language/framework best practices")
        
        return recommendations
    
    @classmethod
    def _calculate_overall_health(cls, quality_metrics: models.CodeQualityMetrics,
                                build_success: bool, tests_passing: bool,
                                security_issues: List[str], performance_issues: List[str]) -> float:
        """Calculate overall project health score"""
        
        # Weight different factors
        weights = {
            'build_success': 0.3,
            'tests_passing': 0.2,
            'quality_metrics': 0.3,
            'security': 0.1,
            'performance': 0.1
        }
        
        score = 0.0
        
        # Build success
        score += weights['build_success'] * (100 if build_success else 0)
        
        # Tests passing
        score += weights['tests_passing'] * (100 if tests_passing else 80)  # 80 if no tests
        
        # Quality metrics (average of all metrics)
        quality_avg = (
            quality_metrics.complexity_score +
            quality_metrics.maintainability_score +
            quality_metrics.test_coverage +
            quality_metrics.security_score +
            quality_metrics.performance_score +
            quality_metrics.best_practices_compliance
        ) / 6
        score += weights['quality_metrics'] * quality_avg
        
        # Security (deduct for issues)
        security_score = max(100 - len(security_issues) * 10, 0)
        score += weights['security'] * security_score
        
        # Performance (deduct for issues)
        performance_score = max(100 - len(performance_issues) * 5, 0)
        score += weights['performance'] * performance_score
        
        return min(score, 100.0) 