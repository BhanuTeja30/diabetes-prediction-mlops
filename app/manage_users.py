"""Command-line user management (works without the API running).

Usage:
    python -m app.manage_users create <username> --role admin   # prompts for the password
    python -m app.manage_users list
    python -m app.manage_users set-role <username> user|admin
    python -m app.manage_users disable <username>
    python -m app.manage_users enable <username>
    python -m app.manage_users reset-password <username>
    python -m app.manage_users delete <username>

In Docker:  docker compose exec api python -m app.manage_users list
"""

from __future__ import annotations

import argparse
import getpass
import sys

from app.auth import AuthError, UserStore
from app.config import get_settings


def _password(prompt: str = "Password: ") -> str:
    first = getpass.getpass(prompt)
    if getpass.getpass("Repeat password: ") != first:
        raise AuthError("Passwords do not match.")
    return first


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create")
    create.add_argument("username")
    create.add_argument("--role", choices=["user", "admin"], default="user")
    sub.add_parser("list")
    role = sub.add_parser("set-role")
    role.add_argument("username")
    role.add_argument("role", choices=["user", "admin"])
    for name in ("disable", "enable", "reset-password", "delete"):
        sub.add_parser(name).add_argument("username")
    args = parser.parse_args(argv)

    store = UserStore(get_settings().auth_db_path)
    try:
        if args.command == "create":
            user = store.create_user(args.username, _password(), role=args.role)
            print(f"Created {user['role']} '{user['username']}'.")
        elif args.command == "list":
            for u in store.list_users():
                state = "active" if u["is_active"] else "disabled"
                print(f"{u['username']:<24} {u['role']:<6} {state:<9} last login: {u['last_login'] or '-'}")
        elif args.command == "set-role":
            print(store.update_user(args.username, role=args.role))
        elif args.command in ("disable", "enable"):
            print(store.update_user(args.username, is_active=args.command == "enable"))
        elif args.command == "reset-password":
            store.update_user(args.username, password=_password("New password: "))
            print(f"Password for '{args.username}' updated.")
        elif args.command == "delete":
            store.delete_user(args.username)
            print(f"Deleted '{args.username}'.")
    except AuthError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
