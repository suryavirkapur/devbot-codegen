# DevBot2 - Enhanced Repository Generation System

An intelligent repository generation system that creates working code projects from Business Requirements Documents (BRDs) with advanced debugging and validation capabilities.

## 🚀 Features

### Core Capabilities
- **Multi-Framework Support**: FastAPI, Actix (Rust), Gin (Go), Elysia (Bun/TypeScript)
- **AI/ML Integration**: PyTorch, Adalflow, Pydantic
- **Database Support**: SQLAlchemy and other ORMs
- **Automatic Dockerfile Generation**: Production-ready multi-stage Docker builds

### Enhanced AI-Powered Workflow
- **Multi-Agent System**: Specialized AI agents for different tasks
- **Smart Debugging Loop**: Automatic error detection and fixing with Docker builds
- **Attempt Tracking**: Comprehensive history of debugging attempts and strategies
- **Validation Agents**: Automated testing and functionality verification
- **Strategy Switching**: Multiple debugging approaches (dependency fixes, Dockerfile optimization, etc.)

## 🏗️ Architecture

The system uses a multi-phase approach:

1. **Generation Phase**: AI agents create project structure and files
2. **Debugging Phase**: Enhanced loop with attempt tracking and error fixing
3. **Validation Phase**: Comprehensive testing and functionality verification

### AI Agent Types
- `structure_generator`: Creates project file structure
- `code_generator`: Generates source code files
- `debugger_*`: Specialized debugging agents (dependency_fix, dockerfile_optimization, etc.)
- `validation_planner`: Creates comprehensive test plans
- `recommendations_generator`: Provides improvement suggestions

## 🛠️ API Endpoints

### Primary Endpoints

#### Generate Repository
```http
POST /api/brds/generate-repo
```

Enhanced repository generation with debugging and validation.

**Request Body:**
```json
{
  "projectName": "My Project",
  "projectDescription": "Description of the project",
  "technologyStack": {
    "frontend": ["React", "TypeScript"],
    "backend": ["FastAPI", "Python"],
    "database": ["PostgreSQL", "SQLAlchemy"],
    "other": ["Docker", "Redis"]
  },
  "coreFeatures": [
    {
      "name": "User Authentication",
      "description": "JWT-based authentication system",
      "priority": "High"
    }
  ],
  "dataModels": [
    {
      "name": "User",
      "fields": "id, email, password_hash, created_at",
      "relationships": "one-to-many with Profile"
    }
  ],
  "authentication": "jwt",
  "apiRequirements": [
    "POST /api/auth/login",
    "GET /api/users/profile"
  ]
}
```

**Response:**
```json
{
  "status_code": 201,
  "message": "Enhanced repository generation completed for 'My Project'",
  "data": {
    "project_name": "My Project",
    "project_path": "/path/to/generated/project",
    "generation_time_seconds": 45.2,
    "validation_summary": {
      "overall_success": true,
      "total_tests": 5,
      "passed_tests": 4,
      "recommendations_count": 1
    },
    "technology_stack": {...},
    "features_count": 3,
    "api_endpoints_count": 2
  }
}
```

#### Get Generation Status
```http
GET /api/brds/generation-status/{project_name}
```

Retrieve detailed information about a generated project.

#### Manual Debugging
```http
POST /api/brds/debug-project
```

Manually trigger debugging for an existing project.

**Request Body:**
```json
{
  "projectName": "My Project",
  "forceRebuild": false
}
```

### Supporting Endpoints

- `POST /api/brds` - Create and improve BRD
- `POST /api/brds/create-from-text` - Generate BRD from text description
- `GET /health` - Health check

## 🔧 Configuration

### Environment Variables

```bash
# Required
OPENAI_API_KEY=your_openai_api_key

# Optional Azure OpenAI
AZURE_OPENAI_ENDPOINT=your_azure_endpoint
AZURE_OPENAI_API_KEY=your_azure_key
AZURE_OPENAI_API_VERSION=your_api_version

# Server
PORT=8080
```

### Dependencies

The system requires:
- Python 3.10+
- Docker (for building and testing generated projects)
- OpenAI API access

## 🚦 Getting Started

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Environment Variables**
   ```bash
   export OPENAI_API_KEY="your-key-here"
   ```

3. **Run the Server**
   ```bash
   python src/app.py
   ```

4. **Generate a Project**
   ```bash
   curl -X POST http://localhost:8080/api/brds/generate-repo \
     -H "Content-Type: application/json" \
     -d '{"projectName": "Test Project", ...}'
   ```

## 🐛 Enhanced Debugging System

### Debug Strategies

The system employs multiple debugging strategies:

1. **dependency_fix**: Package and dependency issues
2. **dockerfile_optimization**: Container build problems
3. **code_compilation**: Source code errors
4. **configuration_fix**: Configuration file issues
5. **multi_stage_analysis**: Comprehensive analysis

### Attempt Tracking

Each debugging session tracks:
- Attempt number and strategy used
- Files modified in each attempt
- Error logs and success status
- Previous attempt summaries for context

### Validation Testing

The validation phase includes:
- Build/compilation verification
- API endpoint testing
- Database connectivity tests
- Integration tests for core features
- Technology-specific test commands

## 📊 Output Structure

Generated projects include:

```
project-name/
├── Dockerfile              # Multi-stage production build
├── requirements.txt        # Or package.json, go.mod, etc.
├── src/                   # Source code
├── tests/                 # Generated tests
├── docs/                  # Documentation
├── validation_report.json # Comprehensive validation results
└── ...                    # Framework-specific files
```

## 🤖 AI Agent Workflow

1. **Structure Generation**: AI creates optimal file structure
2. **Code Generation**: Specialized agents generate each file
3. **Docker Build Testing**: Automated build verification
4. **Smart Debugging**: Multiple AI strategies for error fixing
5. **Validation Planning**: AI creates comprehensive test plans
6. **Test Execution**: Automated testing with timeout handling
7. **Recommendations**: AI-generated improvement suggestions

## 🎯 Supported Technologies

### Backend Frameworks
- **FastAPI** (Python) - High-performance async API framework
- **Actix** (Rust) - High-performance actor framework
- **Gin** (Go) - Lightweight web framework
- **Elysia** (Bun/TypeScript) - Modern web framework

### AI/ML Libraries
- **PyTorch** - Deep learning framework
- **Adalflow** - LLM application framework
- **Pydantic** - Data validation and parsing

### Databases & ORMs
- **SQLAlchemy** (Python)
- **Diesel** (Rust)
- **GORM** (Go)
- **Prisma** (TypeScript)

## 📈 Performance Features

- **Parallel AI Calls**: Multiple agents work simultaneously
- **Intelligent Caching**: Reuse of successful patterns
- **Timeout Management**: Prevents hanging operations
- **Resource Cleanup**: Automatic Docker image cleanup
- **Progress Tracking**: Real-time generation status

## 🛡️ Error Handling

- Comprehensive error logging
- Graceful fallbacks for AI failures
- Validation report generation even on partial failures
- Manual debugging triggers for edge cases
- Detailed error context for troubleshooting

## 📝 License

This project is licensed under the MIT License.

## TODO:

- [x] Implement Python backend using Robyn Web Framework
- [] Figure out how pydantic works
- [] Fix Validations
- [] Code-Debug-Text loop using Docker
