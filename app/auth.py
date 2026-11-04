"""User accounts, password hashing and signed access tokens (standard library only).

Roles:
  * user  - may sign in and request predictions
  * admin - additionally sees model info / monitoring, reloads the model, runs drift
            checks and manages user accounts

Passwords are stored as salted PBKDF2-SHA256 hashes, never in plain text. Accounts hold no
health data, and predictions are never linked to a username.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import secrets
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROLES = ("user", "admin")
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
MIN_PASSWORD_LENGTH = 8
PBKDF2_ITERATIONS = 390_000
MAX_FAILED_LOGINS = 5
LOCKOUT_SECONDS = 300


class AuthError(Exception):
    """Base class; `status` maps to the HTTP status returned by the API."""

    status = 400


class InvalidCredentials(AuthError):
    status = 401


class AccountLocked(AuthError):
    status = 429


class Conflict(AuthError):
    status = 409


class NotFound(AuthError):
    status = 404


# ----------------------------------------------------------------------------- passwords
def hash_password(password: str, iterations: int = PBKDF2_ITERATIONS) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iterations, salt_hex, digest_hex = stored.split("$")
    except ValueError:
        return False
    if algo != "pbkdf2_sha256":
        return False
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iterations))
    return hmac.compare_digest(digest.hex(), digest_hex)


def validate_credentials(username: str, password: str) -> None:
    if not USERNAME_RE.fullmatch(username or ""):
        raise AuthError("Username must be 3-32 characters: letters, digits, '.', '_' or '-'.")
    validate_password(password)


def validate_password(password: str) -> None:
    if len(password or "") < MIN_PASSWORD_LENGTH:
        raise AuthError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")
    if password.isdigit() or password.isalpha():
        raise AuthError("Password must contain both letters and numbers or symbols.")


# ----------------------------------------------------------------------------- tokens
def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


class TokenSigner:
    """Compact HMAC-SHA256 signed tokens: base64(payload).base64(signature)."""

    def __init__(self, secret: str, ttl_seconds: int):
        if len(secret) < 32:
            raise ValueError("AUTH_SECRET_KEY must be at least 32 characters.")
        self._key = secret.encode()
        self.ttl = ttl_seconds

    def issue(self, username: str, role: str, now: float | None = None) -> tuple[str, int]:
        exp = int((now or time.time()) + self.ttl)
        payload = _b64(json.dumps({"sub": username, "role": role, "exp": exp}, separators=(",", ":")).encode())
        sig = _b64(hmac.new(self._key, payload.encode(), hashlib.sha256).digest())
        return f"{payload}.{sig}", exp

    def verify(self, token: str, now: float | None = None) -> dict[str, Any]:
        try:
            payload, sig = token.split(".")
            expected = _b64(hmac.new(self._key, payload.encode(), hashlib.sha256).digest())
            if not hmac.compare_digest(sig, expected):
                raise InvalidCredentials("Invalid token.")
            claims = json.loads(_unb64(payload))
        except (ValueError, json.JSONDecodeError) as exc:
            raise InvalidCredentials("Invalid token.") from exc
        if claims.get("exp", 0) < (now or time.time()):
            raise InvalidCredentials("Session expired. Please sign in again.")
        return claims


# ----------------------------------------------------------------------------- user store
class UserStore:
    """SQLite-backed accounts. Safe for the single-process API (guarded by a lock)."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._failed: dict[str, tuple[int, float]] = {}
        with self._conn() as c:
            c.execute(
                """CREATE TABLE IF NOT EXISTS users (
                       username TEXT PRIMARY KEY COLLATE NOCASE,
                       password_hash TEXT NOT NULL,
                       role TEXT NOT NULL CHECK (role IN ('user', 'admin')),
                       is_active INTEGER NOT NULL DEFAULT 1,
                       created_at TEXT NOT NULL,
                       last_login TEXT
                   )"""
            )

    @contextmanager
    def _conn(self):
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            finally:
                conn.close()

    @staticmethod
    def _public(row: sqlite3.Row) -> dict[str, Any]:
        return {"username": row["username"], "role": row["role"], "is_active": bool(row["is_active"]),
                "created_at": row["created_at"], "last_login": row["last_login"]}

    def create_user(self, username: str, password: str, role: str = "user") -> dict[str, Any]:
        validate_credentials(username, password)
        if role not in ROLES:
            raise AuthError(f"Role must be one of {ROLES}.")
        now = datetime.now(UTC).isoformat(timespec="seconds")
        try:
            with self._conn() as c:
                c.execute("INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
                          (username, hash_password(password), role, now))
        except sqlite3.IntegrityError as exc:
            raise Conflict("That username is already taken.") from exc
        return self.get_user(username)

    def get_user(self, username: str) -> dict[str, Any]:
        with self._conn() as c:
            row = c.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        if row is None:
            raise NotFound(f"User '{username}' not found.")
        return self._public(row)

    def list_users(self) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM users ORDER BY role DESC, username").fetchall()
        return [self._public(r) for r in rows]

    def count_active_admins(self) -> int:
        with self._conn() as c:
            return c.execute("SELECT COUNT(*) FROM users WHERE role = 'admin' AND is_active = 1").fetchone()[0]

    def authenticate(self, username: str, password: str) -> dict[str, Any]:
        key = (username or "").lower()
        fails, since = self._failed.get(key, (0, 0.0))
        if fails >= MAX_FAILED_LOGINS and time.time() - since < LOCKOUT_SECONDS:
            raise AccountLocked("Too many failed sign-in attempts. Try again in a few minutes.")
        with self._conn() as c:
            row = c.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        # Same generic message for unknown user / wrong password / disabled account.
        if row is None or not verify_password(password, row["password_hash"]) or not row["is_active"]:
            self._failed[key] = (fails + 1, time.time() if fails + 1 >= MAX_FAILED_LOGINS else since)
            raise InvalidCredentials("Incorrect username or password, or the account is disabled.")
        self._failed.pop(key, None)
        with self._conn() as c:
            c.execute("UPDATE users SET last_login = ? WHERE username = ?",
                      (datetime.now(UTC).isoformat(timespec="seconds"), row["username"]))
        return self._public(row)

    def update_user(self, username: str, *, role: str | None = None, is_active: bool | None = None,
                    password: str | None = None) -> dict[str, Any]:
        current = self.get_user(username)
        if role is not None and role not in ROLES:
            raise AuthError(f"Role must be one of {ROLES}.")
        losing_admin = current["role"] == "admin" and current["is_active"] and (
            (role is not None and role != "admin") or is_active is False)
        if losing_admin and self.count_active_admins() <= 1:
            raise Conflict("Cannot demote or disable the last active admin.")
        fields, values = [], []
        if role is not None:
            fields.append("role = ?")
            values.append(role)
        if is_active is not None:
            fields.append("is_active = ?")
            values.append(int(is_active))
        if password is not None:
            validate_password(password)
            fields.append("password_hash = ?")
            values.append(hash_password(password))
        if fields:
            with self._conn() as c:
                c.execute(f"UPDATE users SET {', '.join(fields)} WHERE username = ?",  # noqa: S608 - fixed columns
                          (*values, username))
        return self.get_user(username)

    def delete_user(self, username: str) -> None:
        user = self.get_user(username)
        if user["role"] == "admin" and user["is_active"] and self.count_active_admins() <= 1:
            raise Conflict("Cannot delete the last active admin.")
        with self._conn() as c:
            c.execute("DELETE FROM users WHERE username = ?", (username,))

    def ensure_admin(self, username: str | None, password: str | None) -> str | None:
        """Create the bootstrap admin from env vars if no admin exists yet. Returns the name if created."""
        if self.count_active_admins() > 0 or not username or not password:
            return None
        try:
            self.create_user(username, password, role="admin")
        except Conflict:
            self.update_user(username, role="admin", is_active=True, password=password)
        return username
