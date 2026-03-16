"""Unified dashboard: password gate, login, header, phase bar, route registration."""

import os

from fastapi.responses import RedirectResponse
from nicegui import app, ui

from pipeline.config import CHARTER_ELEMENT_IDS
from pipeline.dashboard.shared import N_PHASES, PHASE_ROUTES
from pipeline.phase1.storage import load_annotator_ids

# --- Password middleware ---

PASSWORD = os.environ.get("DASHBOARD_PASSWORD", "")
MAX_PASSWORD_ATTEMPTS = 10

if PASSWORD:
    from fastapi import Request
    from starlette.middleware.base import BaseHTTPMiddleware

    @app.add_middleware
    class PasswordMiddleware(BaseHTTPMiddleware):
        """Redirect all pages to /password if not yet authenticated."""

        async def dispatch(self, request: Request, call_next):
            if not app.storage.user.get("password_ok", False):
                if (
                    not request.url.path.startswith("/_nicegui")
                    and request.url.path != "/password"
                ):
                    return RedirectResponse("/password")
            return await call_next(request)


# --- Shared UI components ---


def render_phase_bar(active_phase: int = 1, right_slot=None):
    """Render a stepper-style phase bar with clickable phase links.

    Args:
        active_phase: Currently active phase number (1-indexed).
        right_slot: Optional callable rendered on the right side of the bar.
    """

    def _circle(n: int) -> str:
        style = (
            "background:#1976d2;border:2px solid #1976d2;color:white;"
            if n <= active_phase
            else "background:transparent;border:2px solid #555;color:#666;"
        )
        return (
            f'<div style="width:26px;height:26px;border-radius:50%;{style}'
            f"display:flex;align-items:center;justify-content:center;"
            f'font-size:0.8em;font-weight:600;flex-shrink:0;">{n}</div>'
        )

    def _label(n: int) -> str:
        color = "white" if n == active_phase else "#666"
        weight = "600" if n == active_phase else "400"
        return (
            f'<span style="color:{color};font-size:0.8em;font-weight:{weight};'
            f'white-space:nowrap;margin-left:6px;">Phase {n}</span>'
        )

    connector_color = lambda n: "#1976d2" if n < active_phase else "#444"
    connector = (
        lambda n: f'<div style="width:40px;height:2px;background:{connector_color(n)};margin:0 8px;flex-shrink:0;"></div>'
    )

    parts = []
    for n in range(1, N_PHASES + 1):
        if n > 1:
            parts.append(connector(n - 1))
        route = PHASE_ROUTES.get(n, "#")
        parts.append(
            f'<a href="{route}" style="text-decoration:none;display:flex;align-items:center;">'
            + _circle(n)
            + _label(n)
            + "</a>"
        )

    stepper_html = (
        '<div style="display:flex;align-items:center;padding:6px 0;">'
        + "".join(parts)
        + "</div>"
    )

    with (
        ui.row()
        .classes("items-center justify-between w-full q-px-md")
        .style("background:#252525;border-top:1px solid #333;min-height:44px;")
    ):
        ui.html(stepper_html)
        if right_slot:
            with ui.row().classes("items-center gap-2"):
                right_slot()


def render_header(annotator_id: str, active_phase: int = 1, right_slot=None):
    """Render the shared page header: title bar + phase stepper.

    Args:
        annotator_id: Current user's name (empty string if not logged in).
        active_phase: Currently active phase number, passed through to render_phase_bar.
        right_slot: Optional callable for phase-bar right side (phase-specific actions).
    """
    with (
        ui.header()
        .classes("column items-stretch q-pa-none")
        .style("background: #1d1d1d;")
    ):
        with ui.row().classes("items-center justify-between q-px-md q-py-xs w-full"):
            ui.label("Model Raising Annotation Platform").classes("text-h6 text-white")
            with ui.row().classes("items-center gap-4"):
                if annotator_id:
                    ui.label(f"Account: {annotator_id}").classes(
                        "text-caption text-weight-medium"
                    ).style("color:#aaa;")
                ui.button(
                    "Logout",
                    on_click=lambda: (
                        app.storage.user.clear(),
                        ui.navigate.to("/"),
                    ),
                ).classes("text-white").props("flat dense")
        render_phase_bar(active_phase, right_slot=right_slot)


# --- Shared pages ---


@ui.page("/password")
def password_page():
    """Simple password gate, stored in user storage (cookie-persisted)."""
    if not PASSWORD or app.storage.user.get("password_ok", False):
        return RedirectResponse("/")

    def try_password():
        attempts = app.storage.user.get("password_attempts", 0)
        if attempts >= MAX_PASSWORD_ATTEMPTS:
            ui.notify("Too many failed attempts. Access locked.", color="negative")
            return
        if pw_input.value == PASSWORD:
            app.storage.user["password_ok"] = True
            app.storage.user["password_attempts"] = 0
            ui.navigate.to("/")
        else:
            attempts += 1
            app.storage.user["password_attempts"] = attempts
            remaining = MAX_PASSWORD_ATTEMPTS - attempts
            ui.notify(
                f"Wrong password. {remaining} attempt(s) remaining.", color="negative"
            )
            pw_input.set_value("")

    with ui.column().classes("absolute-center items-center gap-4"):
        ui.label("Model Raising Annotation Platform").classes(
            "text-h4 text-weight-bold"
        )
        ui.label("Enter the password to continue.").classes(
            "text-subtitle1 text-grey-7"
        )
        pw_input = (
            ui.input("Password", password=True, password_toggle_button=True)
            .on("keydown.enter", try_password)
            .classes("w-64")
        )
        ui.button("Enter", on_click=try_password, color="primary").classes("w-64")
    return None


@ui.page("/")
def login_page():
    """Login page where annotator enters their name."""
    existing_names = load_annotator_ids()

    with ui.column().classes("absolute-center items-center gap-4"):
        ui.label("Model Raising Annotation Platform").classes(
            "text-h4 text-weight-bold"
        )
        ui.label("Enter your name to begin annotating.").classes(
            "text-subtitle1 text-grey-7"
        )

        name_input = ui.input(
            label="Annotator name",
            placeholder="e.g. Alice",
            autocomplete=existing_names,
        ).classes("w-64")

        def start():
            val = name_input.value
            if not val or not str(val).strip():
                ui.notify("Please enter a name", type="warning")
                return
            app.storage.user["annotator_id"] = str(val).strip()
            ui.navigate.to("/annotate")

        name_input.on("keydown.enter", lambda _: start())
        ui.button("Start annotating", on_click=start, color="primary").classes("w-64")


# --- Register phase routes (import triggers @ui.page decorators) ---
import pipeline.dashboard.phase1  # noqa: F401, E402
import pipeline.dashboard.phase2  # noqa: F401, E402
import pipeline.dashboard.phase3  # noqa: F401, E402
