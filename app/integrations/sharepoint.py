"""
Write a feedback row to a SharePoint Excel file via Microsoft Graph API.

Required .env keys:
  AZURE_TENANT_ID      – your AAD tenant ID
  AZURE_CLIENT_ID      – app registration client ID
  AZURE_CLIENT_SECRET  – app registration client secret
  SHAREPOINT_EXCEL_URL – the sharing URL of the Excel file
  SHAREPOINT_SHEET     – worksheet name (default: Sheet1)
  SHAREPOINT_TABLE     – table name inside the sheet (default: FeedbackTable)

The Excel table must already exist with columns:
  Timestamp | User | Type | Rating | Message | Page
"""
import base64
import logging
import os
from typing import Optional

logger = logging.getLogger("autoeda.integrations.sharepoint")


def _get_token(tenant_id: str, client_id: str, client_secret: str) -> Optional[str]:
    import urllib.request, urllib.parse, json
    url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
    data = urllib.parse.urlencode({
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
        "scope": "https://graph.microsoft.com/.default",
    }).encode()
    try:
        req = urllib.request.Request(url, data=data, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())["access_token"]
    except Exception as e:
        logger.warning("SharePoint token error: %s", e)
        return None


def _encode_sharing_url(url: str) -> str:
    b64 = base64.urlsafe_b64encode(url.encode()).rstrip(b"=").decode()
    return "u!" + b64


def _get_drive_item(token: str, sharing_url: str) -> Optional[dict]:
    import urllib.request, json
    encoded = _encode_sharing_url(sharing_url)
    url = f"https://graph.microsoft.com/v1.0/shares/{encoded}/driveItem"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except Exception as e:
        logger.warning("SharePoint driveItem error: %s", e)
        return None


def append_feedback_row(
    timestamp: str,
    user: str,
    feedback_type: str,
    rating: str,
    message: str,
    page: str,
) -> bool:
    """Append one row to the SharePoint Excel feedback table. Returns True on success."""
    from ..config import settings

    tenant_id     = getattr(settings, "AZURE_TENANT_ID", "") or os.environ.get("AZURE_TENANT_ID", "")
    client_id     = getattr(settings, "AZURE_CLIENT_ID", "") or os.environ.get("AZURE_CLIENT_ID", "")
    client_secret = getattr(settings, "AZURE_CLIENT_SECRET", "") or os.environ.get("AZURE_CLIENT_SECRET", "")
    excel_url     = getattr(settings, "SHAREPOINT_EXCEL_URL", "") or os.environ.get("SHAREPOINT_EXCEL_URL", "")
    sheet         = getattr(settings, "SHAREPOINT_SHEET", "Sheet1") or os.environ.get("SHAREPOINT_SHEET", "Sheet1")
    table         = getattr(settings, "SHAREPOINT_TABLE", "FeedbackTable") or os.environ.get("SHAREPOINT_TABLE", "FeedbackTable")

    if not all([tenant_id, client_id, client_secret, excel_url]):
        logger.info("SharePoint not configured — skipping Excel write")
        return False

    token = _get_token(tenant_id, client_id, client_secret)
    if not token:
        return False

    item = _get_drive_item(token, excel_url)
    if not item:
        return False

    drive_id = item.get("parentReference", {}).get("driveId")
    item_id  = item.get("id")
    if not drive_id or not item_id:
        logger.warning("Could not resolve driveId/itemId from SharePoint item")
        return False

    import urllib.request, json
    url = (
        f"https://graph.microsoft.com/v1.0/drives/{drive_id}/items/{item_id}"
        f"/workbook/worksheets/{sheet}/tables/{table}/rows/add"
    )
    payload = json.dumps({"values": [[timestamp, user, feedback_type, rating, message, page]]}).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            logger.info("SharePoint feedback row added (status %s)", resp.status)
            return True
    except Exception as e:
        logger.warning("SharePoint append error: %s", e)
        return False
