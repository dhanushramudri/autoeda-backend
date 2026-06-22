"""Dataset Library: a free-for-all, cross-workspace wiki documenting what
datasets are for (business use case, project use case, etc.), organized by
theme (Churn, Forecasting, Revenue Prediction, ...). Any authenticated user
can create categories/articles, edit any article, and attach files. Linking
an article to a dataset never grants new access to that dataset — viewers
still need real workspace membership (or the dataset must be globally
shared) to see its details or download it.
"""
import hashlib
import json
import re
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session

from ..auth import get_current_active_user
from ..dataset_access import accessible_datasets_query, has_dataset_access
from ..database import get_db
from ..models.dataset import Dataset
from ..models.dataset_doc import (
    DocArticle, DocArticleDataset, DocAttachment, DocAttachmentUpload, DocCategory,
)
from ..models.job import BackgroundJob
from ..models.user import User
from ..models.workspace import WorkspaceMember
from ..s3_attachments import (
    delete_object, get_object_bytes, head_object, new_object_key, presign_get, presign_put, put_object_bytes,
)
from ..schemas.dataset import DatasetCreateResponse
from ..schemas.dataset_doc import (
    AttachmentConfirmRequest,
    AttachmentPresignRequest,
    AttachmentPresignResponse,
    DocArticleCreate,
    DocArticleListItem,
    DocArticleResponse,
    DocArticleUpdate,
    DocAttachmentResponse,
    DocCategoryCreate,
    DocCategoryResponse,
    ImportAttachmentRequest,
    LinkedDataset,
)

router = APIRouter(tags=["dataset_docs"])


def _user_name(db: Session, user_id: int | None) -> str | None:
    if user_id is None:
        return None
    u = db.query(User).filter(User.id == user_id).first()
    return u.full_name if u else None


def _linked_datasets(db: Session, article_id: int, viewer: User) -> list[LinkedDataset]:
    links = db.query(DocArticleDataset).filter(DocArticleDataset.article_id == article_id).all()
    out = []
    for link in links:
        ds = db.query(Dataset).filter(Dataset.id == link.dataset_id).first()
        if ds and has_dataset_access(ds, viewer, db):
            out.append(LinkedDataset(
                id=ds.id, name=ds.name, row_count=ds.row_count,
                column_count=ds.column_count, workspace_id=ds.workspace_id,
            ))
    return out


def _content_preview(html: str, length: int = 160) -> str:
    text = re.sub(r"<[^>]*>", " ", html or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:length]


def _article_list_item(db: Session, a: DocArticle, viewer: User) -> DocArticleListItem:
    attachment_count = db.query(DocAttachment).filter(DocAttachment.article_id == a.id).count()
    return DocArticleListItem(
        id=a.id, category_id=a.category_id, title=a.title, summary=a.summary,
        content_preview=_content_preview(a.content),
        created_by=a.created_by, created_by_name=_user_name(db, a.created_by),
        created_at=a.created_at,
        updated_by=a.updated_by, updated_by_name=_user_name(db, a.updated_by),
        updated_at=a.updated_at,
        datasets=_linked_datasets(db, a.id, viewer),
        attachment_count=attachment_count,
    )


# ── Categories ───────────────────────────────────────────────────────────────

@router.get("/doc-categories", response_model=list[DocCategoryResponse])
def list_categories(db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    categories = db.query(DocCategory).order_by(DocCategory.name).all()
    out = []
    for c in categories:
        count = db.query(DocArticle).filter(DocArticle.category_id == c.id).count()
        out.append(DocCategoryResponse(
            id=c.id, name=c.name, description=c.description,
            created_by=c.created_by, created_at=c.created_at, article_count=count,
        ))
    return out


@router.post("/doc-categories", response_model=DocCategoryResponse, status_code=201)
def create_category(
    body: DocCategoryCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    existing = db.query(DocCategory).filter(DocCategory.name.ilike(body.name.strip())).first()
    if existing:
        raise HTTPException(status_code=409, detail="A category with this name already exists")
    cat = DocCategory(name=body.name.strip(), description=body.description, created_by=current_user.id)
    db.add(cat)
    db.commit()
    db.refresh(cat)
    return DocCategoryResponse(
        id=cat.id, name=cat.name, description=cat.description,
        created_by=cat.created_by, created_at=cat.created_at, article_count=0,
    )


@router.delete("/doc-categories/{category_id}", status_code=204)
def delete_category(
    category_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    cat = db.query(DocCategory).filter(DocCategory.id == category_id).first()
    if not cat:
        raise HTTPException(status_code=404, detail="Category not found")
    if not current_user.is_admin and cat.created_by != current_user.id:
        raise HTTPException(status_code=403, detail="Only the creator or an admin can delete this category")
    db.delete(cat)
    db.commit()


# ── Articles ─────────────────────────────────────────────────────────────────

@router.get("/doc-categories/{category_id}/articles", response_model=list[DocArticleListItem])
def list_articles(
    category_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    articles = (
        db.query(DocArticle)
        .filter(DocArticle.category_id == category_id)
        .order_by(DocArticle.updated_at.desc())
        .all()
    )
    return [_article_list_item(db, a, current_user) for a in articles]


@router.get("/doc-articles/{article_id}", response_model=DocArticleResponse)
def get_article(
    article_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    a = db.query(DocArticle).filter(DocArticle.id == article_id).first()
    if not a:
        raise HTTPException(status_code=404, detail="Article not found")
    attachments = db.query(DocAttachment).filter(DocAttachment.article_id == a.id).all()
    return DocArticleResponse(
        id=a.id, category_id=a.category_id, title=a.title, summary=a.summary, content=a.content,
        created_by=a.created_by, created_by_name=_user_name(db, a.created_by),
        created_at=a.created_at,
        updated_by=a.updated_by, updated_by_name=_user_name(db, a.updated_by),
        updated_at=a.updated_at,
        datasets=_linked_datasets(db, a.id, current_user),
        attachment_count=len(attachments),
        attachments=[
            DocAttachmentResponse(
                id=att.id, article_id=att.article_id, filename=att.filename,
                content_type=att.content_type, file_size_bytes=att.file_size_bytes,
                uploaded_by=att.uploaded_by, uploaded_by_name=_user_name(db, att.uploaded_by),
                uploaded_at=att.uploaded_at,
            ) for att in attachments
        ],
    )


def _set_dataset_links(db: Session, article_id: int, dataset_ids: list[int], user: User) -> None:
    db.query(DocArticleDataset).filter(DocArticleDataset.article_id == article_id).delete()
    seen = set()
    for did in dataset_ids:
        if did in seen:
            continue
        seen.add(did)
        ds = db.query(Dataset).filter(Dataset.id == did).first()
        if not ds or not has_dataset_access(ds, user, db):
            continue  # silently skip datasets the author can't actually access
        db.add(DocArticleDataset(article_id=article_id, dataset_id=did))


@router.post("/doc-articles", response_model=DocArticleResponse, status_code=201)
def create_article(
    payload: DocArticleCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    cat = db.query(DocCategory).filter(DocCategory.id == payload.category_id).first()
    if not cat:
        raise HTTPException(status_code=404, detail="Category not found")

    a = DocArticle(
        category_id=payload.category_id, title=payload.title.strip(),
        summary=payload.summary, content=payload.content,
        created_by=current_user.id, updated_by=current_user.id,
    )
    db.add(a)
    db.flush()
    _set_dataset_links(db, a.id, payload.dataset_ids, current_user)
    db.commit()
    db.refresh(a)
    return get_article(a.id, db, current_user)


@router.patch("/doc-articles/{article_id}", response_model=DocArticleResponse)
def update_article(
    article_id: int,
    payload: DocArticleUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    a = db.query(DocArticle).filter(DocArticle.id == article_id).first()
    if not a:
        raise HTTPException(status_code=404, detail="Article not found")

    if payload.category_id is not None:
        if not db.query(DocCategory).filter(DocCategory.id == payload.category_id).first():
            raise HTTPException(status_code=404, detail="Category not found")
        a.category_id = payload.category_id
    if payload.title is not None:
        a.title = payload.title.strip()
    if payload.summary is not None:
        a.summary = payload.summary
    if payload.content is not None:
        a.content = payload.content
    if payload.dataset_ids is not None:
        _set_dataset_links(db, a.id, payload.dataset_ids, current_user)
    a.updated_by = current_user.id

    db.commit()
    db.refresh(a)
    return get_article(a.id, db, current_user)


@router.delete("/doc-articles/{article_id}", status_code=204)
def delete_article(
    article_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    a = db.query(DocArticle).filter(DocArticle.id == article_id).first()
    if not a:
        raise HTTPException(status_code=404, detail="Article not found")
    if not current_user.is_admin and a.created_by != current_user.id:
        raise HTTPException(status_code=403, detail="Only the author or an admin can delete this article")
    db.delete(a)
    db.commit()


# ── Attachments ──────────────────────────────────────────────────────────────
#
# Uploads go straight from the browser to S3 via a presigned URL (init here →
# PUT to S3 → confirm here), never through this API's own request body — the
# Vercel proxy in front of this backend caps bodies at ~4.5MB, which a 100MB
# dataset attachment would blow straight through.

MAX_ATTACHMENT_BYTES = 100 * 1024 * 1024  # 100MB


@router.post("/doc-articles/{article_id}/attachments/presign", response_model=AttachmentPresignResponse, status_code=201)
def presign_attachment_upload(
    article_id: int,
    body: AttachmentPresignRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    a = db.query(DocArticle).filter(DocArticle.id == article_id).first()
    if not a:
        raise HTTPException(status_code=404, detail="Article not found")
    if body.file_size_bytes > MAX_ATTACHMENT_BYTES:
        raise HTTPException(status_code=413, detail="Attachment exceeds 100MB limit")
    if body.file_size_bytes <= 0:
        raise HTTPException(status_code=400, detail="Empty file")

    key = new_object_key(article_id, body.filename or "attachment")
    pending = DocAttachmentUpload(
        article_id=article_id, s3_key=key, filename=body.filename or "attachment",
        content_type=body.content_type, expected_size_bytes=body.file_size_bytes,
        created_by=current_user.id,
    )
    db.add(pending)
    db.commit()
    db.refresh(pending)

    return AttachmentPresignResponse(
        upload_id=pending.id,
        upload_url=presign_put(key, body.content_type),
    )


@router.post("/doc-articles/{article_id}/attachments/confirm", response_model=DocAttachmentResponse, status_code=201)
def confirm_attachment_upload(
    article_id: int,
    body: AttachmentConfirmRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    pending = (
        db.query(DocAttachmentUpload)
        .filter(DocAttachmentUpload.id == body.upload_id, DocAttachmentUpload.article_id == article_id)
        .first()
    )
    if not pending:
        raise HTTPException(status_code=404, detail="Upload not found")

    meta = head_object(pending.s3_key)
    if not meta:
        raise HTTPException(status_code=400, detail="File not found in storage — upload may have failed")
    actual_size = meta.get("ContentLength", 0)
    if actual_size != pending.expected_size_bytes:
        delete_object(pending.s3_key)
        db.delete(pending)
        db.commit()
        raise HTTPException(status_code=400, detail="Uploaded file size doesn't match — please retry")

    att = DocAttachment(
        article_id=article_id, filename=pending.filename, content_type=pending.content_type,
        s3_key=pending.s3_key, file_size_bytes=actual_size, uploaded_by=current_user.id,
    )
    db.add(att)
    db.delete(pending)
    db.commit()
    db.refresh(att)
    return DocAttachmentResponse(
        id=att.id, article_id=att.article_id, filename=att.filename,
        content_type=att.content_type, file_size_bytes=att.file_size_bytes,
        uploaded_by=att.uploaded_by, uploaded_by_name=_user_name(db, att.uploaded_by),
        uploaded_at=att.uploaded_at,
    )


@router.get("/doc-attachments/{attachment_id}/download")
def download_attachment(
    attachment_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Always returns JSON {download_url}, not a redirect: the Vercel proxy in
    front of this API follows redirects itself (server-side fetch), which
    would pull the whole file through the proxy again. Returning the URL as
    data lets the browser navigate to S3 directly instead.

    Attachments from before S3 support (bytes still in file_data) are lazily
    migrated to S3 on first download, so every attachment ends up on the same
    code path — no separate streaming branch to maintain."""
    att = db.query(DocAttachment).filter(DocAttachment.id == attachment_id).first()
    if not att:
        raise HTTPException(status_code=404, detail="Attachment not found")
    if not att.s3_key:
        key = new_object_key(att.article_id, att.filename)
        put_object_bytes(key, att.file_data, att.content_type)
        att.s3_key = key
        att.file_data = None
        db.commit()
    return {"download_url": presign_get(att.s3_key, att.filename)}


@router.delete("/doc-attachments/{attachment_id}", status_code=204)
def delete_attachment(
    attachment_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    att = db.query(DocAttachment).filter(DocAttachment.id == attachment_id).first()
    if not att:
        raise HTTPException(status_code=404, detail="Attachment not found")
    if not current_user.is_admin and att.uploaded_by != current_user.id:
        raise HTTPException(status_code=403, detail="Only the uploader or an admin can remove this attachment")
    if att.s3_key:
        delete_object(att.s3_key)
    db.delete(att)
    db.commit()


@router.post(
    "/doc-attachments/{attachment_id}/import",
    response_model=DatasetCreateResponse,
    status_code=201,
)
def import_attachment_to_workspace(
    attachment_id: int,
    payload: ImportAttachmentRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Copy a raw file attachment (e.g. a CSV someone attached to an article)
    into one of the current user's workspaces as a real Dataset, with full EDA."""
    att = db.query(DocAttachment).filter(DocAttachment.id == attachment_id).first()
    if not att:
        raise HTTPException(status_code=404, detail="Attachment not found")

    if not current_user.is_admin:
        member = db.query(WorkspaceMember).filter(
            WorkspaceMember.workspace_id == payload.workspace_id,
            WorkspaceMember.user_id == current_user.id,
        ).first()
        if not member or member.role not in ("admin", "analyst"):
            raise HTTPException(status_code=403, detail="Not authorized to add datasets to that workspace")

    content = get_object_bytes(att.s3_key) if att.s3_key else att.file_data
    if not content:
        raise HTTPException(status_code=400, detail="Could not read this attachment's file data")

    name = (payload.name or "").strip() or re.sub(r"\.[^.]+$", "", att.filename) or att.filename

    ds = Dataset(
        workspace_id=payload.workspace_id,
        name=name,
        source_type="file",
        source_config="{}",
        status="processing",
        created_by=current_user.id,
        file_data=content,
        content_hash=hashlib.md5(content).hexdigest(),
        file_size_bytes=len(content),
        file_path=att.filename,
    )
    db.add(ds)
    db.commit()
    db.refresh(ds)

    job_id = str(uuid.uuid4())
    job = BackgroundJob(
        id=job_id, job_type="eda_pipeline", status="pending", progress=0,
        dataset_id=ds.id, created_by=current_user.id,
    )
    db.add(job)
    db.commit()

    from ..tasks import run_eda_pipeline
    background_tasks.add_task(run_eda_pipeline, job_id, ds.id, None, {})

    resp = DatasetCreateResponse.model_validate(ds)
    resp.job_id = job_id
    return resp


# ── Dataset picker + reverse lookup ────────────────────────────────────────

@router.get("/doc-dataset-search")
def search_accessible_datasets(
    q: str | None = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Datasets the current user can actually access, across every workspace
    they belong to — for the article editor's "link a dataset" picker."""
    query = accessible_datasets_query(db, current_user)
    if q:
        query = query.filter(Dataset.name.ilike(f"%{q}%"))
    results = query.order_by(Dataset.name).limit(50).all()
    return [
        {"id": d.id, "name": d.name, "workspace_id": d.workspace_id, "row_count": d.row_count, "column_count": d.column_count}
        for d in results
    ]


@router.get("/datasets/{dataset_id}/doc-articles", response_model=list[DocArticleListItem])
def get_articles_for_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Reverse lookup for a dataset's own page: which articles document it."""
    links = db.query(DocArticleDataset).filter(DocArticleDataset.dataset_id == dataset_id).all()
    articles = [db.query(DocArticle).filter(DocArticle.id == link.article_id).first() for link in links]
    return [_article_list_item(db, a, current_user) for a in articles if a]
