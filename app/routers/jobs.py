from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..auth import get_current_active_user
from ..database import get_db
from ..models.job import BackgroundJob
from ..models.user import User
from ..schemas.eda import JobStatus

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/{job_id}", response_model=JobStatus)
def get_job_status(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    job = db.query(BackgroundJob).filter(BackgroundJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    import json
    result_data = json.loads(job.result_data) if job.result_data else None

    return JobStatus(
        job_id=job.id,
        status=job.status,
        progress=job.progress,
        message=job.message,
        result_data=result_data,
    )


@router.get("/", response_model=list[JobStatus])
def list_jobs(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    jobs = (
        db.query(BackgroundJob)
        .filter(BackgroundJob.created_by == current_user.id)
        .order_by(BackgroundJob.created_at.desc())
        .limit(50)
        .all()
    )
    import json
    return [
        JobStatus(
            job_id=j.id,
            status=j.status,
            progress=j.progress,
            message=j.message,
            result_data=json.loads(j.result_data) if j.result_data else None,
        )
        for j in jobs
    ]
