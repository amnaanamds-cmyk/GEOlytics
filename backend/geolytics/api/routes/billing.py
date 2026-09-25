"""Billing endpoints: checkout, customer portal and the Stripe webhook."""

from __future__ import annotations

import json
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from geolytics.api.deps import AppSettings, CurrentOrg, DbSession, require_role
from geolytics.api.schemas_auth import CheckoutRequest, CheckoutSession
from geolytics.billing import (
    BillingError,
    SignatureError,
    apply_subscription,
    build_client,
    parse_event,
    verify_webhook_signature,
)
from geolytics.db.models import User
from geolytics.tenancy.principal import Principal

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/billing", tags=["billing"])

RequireOwner = Annotated[Principal, Depends(require_role("owner"))]

# Stripe retries anything non-2xx, so a body we cannot act on must still be
# acknowledged once it is proven authentic.
_ACKNOWLEDGED = Response(status_code=status.HTTP_200_OK)


@router.post("/checkout", response_model=CheckoutSession)
def create_checkout(
    payload: CheckoutRequest,
    session: DbSession,
    settings: AppSettings,
    org: CurrentOrg,
    principal: RequireOwner,
) -> CheckoutSession:
    """Start a Stripe Checkout session for a plan upgrade."""
    if not settings.billing_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="billing is not enabled on this deployment",
        )

    price_id = settings.stripe_price_ids.get(payload.plan)
    if not price_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"no price configured for plan {payload.plan!r}",
        )

    user = session.get(User, principal.user_id) if principal.user_id else None
    try:
        result = build_client(settings).create_checkout_session(
            customer_id=org.billing_customer_id,
            price_id=price_id,
            org_id=org.id,
            success_url=payload.success_url,
            cancel_url=payload.cancel_url,
            customer_email=user.email if user else None,
        )
    except BillingError as exc:
        logger.exception("checkout failed", extra={"org_id": org.id})
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc

    return CheckoutSession(url=result.url, session_id=result.session_id)


@router.post("/portal", response_model=CheckoutSession)
def create_portal(
    settings: AppSettings,
    org: CurrentOrg,
    principal: RequireOwner,
    return_url: str = "https://app.geolytics.example/settings/billing",
) -> CheckoutSession:
    if not org.billing_customer_id:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="this organisation has no billing account yet",
        )
    url = build_client(settings).create_billing_portal_session(
        customer_id=org.billing_customer_id, return_url=return_url
    )
    return CheckoutSession(url=url, session_id="portal")


@router.post("/webhook", include_in_schema=False)
async def stripe_webhook(
    request: Request, session: DbSession, settings: AppSettings
) -> Response:
    """Receive Stripe events.

    Unauthenticated by necessity -- Stripe cannot hold a credential -- so the
    signature *is* the authentication. Nothing in the body is read until it
    verifies.
    """
    payload = await request.body()
    try:
        verify_webhook_signature(
            payload,
            request.headers.get("stripe-signature", ""),
            settings.stripe_webhook_secret or "",
        )
    except SignatureError as exc:
        # 400, not 401: a bad signature is a malformed request from Stripe's
        # point of view, and a 401 would make Stripe retry it forever.
        logger.warning("rejected stripe webhook", extra={"reason": str(exc)})
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="invalid signature"
        ) from exc

    try:
        event = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="malformed event body"
        ) from exc

    state = parse_event(event, settings.stripe_price_ids)
    if state is None:
        return _ACKNOWLEDGED

    obj = (event.get("data") or {}).get("object") or {}
    reference = obj.get("client_reference_id") or (obj.get("metadata") or {}).get("org_id")
    org_id = int(reference) if isinstance(reference, str) and reference.isdigit() else None

    org = apply_subscription(session, state, org_id=org_id)
    if org is not None:
        logger.info(
            "subscription applied",
            extra={"org_id": org.id, "plan": org.plan, "status": org.status},
        )
    return _ACKNOWLEDGED


@router.get("/subscription")
def read_subscription(org: CurrentOrg, principal: RequireOwner) -> dict[str, object]:
    return {
        "plan": org.plan,
        "status": org.status,
        "customer_id": org.billing_customer_id,
        "subscription_id": org.billing_subscription_id,
        "current_period_end": org.current_period_end,
    }
