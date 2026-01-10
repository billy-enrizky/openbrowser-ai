"""API routes for projects."""

import logging
from datetime import datetime
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query

from app.models.schemas import (
    CreateProjectRequest,
    ProjectListResponse,
    ProjectResponse,
    UpdateProjectRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/projects", tags=["projects"])

# In-memory storage for demo (use database in production)
_projects: dict[str, dict] = {}


@router.post("", response_model=ProjectResponse)
async def create_project(request: CreateProjectRequest):
    """Create a new project."""
    project_id = str(uuid4())
    now = datetime.utcnow()
    
    project = {
        "id": project_id,
        "name": request.name,
        "description": request.description,
        "created_at": now,
        "updated_at": now,
        "task_count": 0,
    }
    _projects[project_id] = project
    
    return ProjectResponse(**project)


@router.get("", response_model=ProjectListResponse)
async def list_projects(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
):
    """List all projects."""
    projects = list(_projects.values())
    
    # Paginate
    start = (page - 1) * page_size
    end = start + page_size
    paginated = projects[start:end]
    
    return ProjectListResponse(
        projects=[ProjectResponse(**p) for p in paginated],
        total=len(projects),
    )


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(project_id: str):
    """Get a specific project."""
    if project_id not in _projects:
        raise HTTPException(status_code=404, detail="Project not found")
    
    return ProjectResponse(**_projects[project_id])


@router.patch("/{project_id}", response_model=ProjectResponse)
async def update_project(project_id: str, request: UpdateProjectRequest):
    """Update a project."""
    if project_id not in _projects:
        raise HTTPException(status_code=404, detail="Project not found")
    
    project = _projects[project_id]
    
    if request.name is not None:
        project["name"] = request.name
    if request.description is not None:
        project["description"] = request.description
    
    project["updated_at"] = datetime.utcnow()
    
    return ProjectResponse(**project)


@router.delete("/{project_id}")
async def delete_project(project_id: str):
    """Delete a project."""
    if project_id not in _projects:
        raise HTTPException(status_code=404, detail="Project not found")
    
    del _projects[project_id]
    return {"status": "deleted", "project_id": project_id}

