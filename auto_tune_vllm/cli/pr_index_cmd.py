"""CLI commands for the Git-LFS-distributed vLLM pull request index."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from ..knowledge.vllm_pr_index import DEFAULT_DATABASE_PATH, VllmPullRequestIndex

console = Console()
app = typer.Typer(help="Synchronize and search the local vLLM merged-PR index.")


def _index(
    database: Path, repository: str, token: Optional[str]
) -> VllmPullRequestIndex:
    return VllmPullRequestIndex(
        database_path=database,
        repository=repository,
        token=token or os.environ.get("GITHUB_TOKEN"),
    )


@app.command("sync")
def sync_command(
    database: Path = typer.Option(
        DEFAULT_DATABASE_PATH, "--database", help="SQLite index path"
    ),
    manifest: Optional[Path] = typer.Option(
        None, "--manifest", help="Write provenance JSON here"
    ),
    repository: str = typer.Option(
        "vllm-project/vllm", "--repository", help="GitHub owner/repository"
    ),
    token: Optional[str] = typer.Option(
        None, "--token", envvar="GITHUB_TOKEN", help="Read-only GitHub token"
    ),
    max_pages: Optional[int] = typer.Option(
        None, "--max-pages", min=1, help="Limit API pages for this sync"
    ),
    metadata_only: bool = typer.Option(
        False,
        "--metadata-only",
        help="Skip per-PR changed-file requests for a fast initial import",
    ),
) -> None:
    """Incrementally fetch merged pull requests from GitHub."""
    index = _index(database, repository, token)
    try:
        result = index.sync(max_pages=max_pages, include_files=not metadata_only)
    except Exception as exc:
        console.print(f"[red]Index sync failed: {exc}[/red]")
        raise typer.Exit(1) from exc
    manifest_path = manifest or database.with_suffix(".manifest.json")
    index.write_manifest(manifest_path)
    console.print(
        f"Indexed {result['indexed']} merged PRs (scanned {result['scanned']})."
    )
    console.print(f"Database: {database}; manifest: {manifest_path}")


@app.command("search")
def search_command(
    query: str = typer.Argument("", help="Free-text search query"),
    database: Path = typer.Option(
        DEFAULT_DATABASE_PATH, "--database", help="SQLite index path"
    ),
    architecture: Optional[str] = typer.Option(
        None, "--architecture", help="Derived architecture facet"
    ),
    hardware: Optional[str] = typer.Option(
        None, "--hardware", help="Derived hardware facet"
    ),
    limit: int = typer.Option(10, "--limit", min=1, max=100, help="Maximum results"),
    json_output: bool = typer.Option(
        False, "--json", help="Print machine-readable results"
    ),
) -> None:
    """Search a locally materialized index without GitHub access."""
    if not database.exists():
        console.print(
            f"[red]Index not found: {database}. "
            "Run 'pr-index sync' or git lfs pull.[/red]"
        )
        raise typer.Exit(1)
    results = VllmPullRequestIndex(database).search(
        query, architecture, hardware, limit
    )
    if json_output:
        console.print_json(json.dumps([result.__dict__ for result in results]))
        return
    table = Table(title="vLLM merged pull requests")
    table.add_column("PR", style="cyan", no_wrap=True)
    table.add_column("Merged", style="dim", no_wrap=True)
    table.add_column("Title", style="white")
    table.add_column("Facets", style="green")
    for result in results:
        facets = ", ".join(item for values in result.facets.values() for item in values)
        table.add_row(f"#{result.number}", result.merged_at[:10], result.title, facets)
    console.print(table)
    for result in results:
        console.print(f"[link={result.url}]{result.url}[/link]\n  {result.excerpt}")
