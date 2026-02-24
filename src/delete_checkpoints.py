"""
Delete tinker checkpoints containing a given string in their name.
"""

import argparse
import tinker
from rich.console import Console
from rich.table import Table

console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="Delete tinker checkpoints whose path or ID contains a given string"
    )
    parser.add_argument(
        "filter",
        type=str,
        help="substring to match against checkpoint path/ID (case-insensitive)",
    )
    args = parser.parse_args()
    filter_str = args.filter.lower()

    service_client = tinker.ServiceClient()
    rest_client = service_client.create_rest_client()

    console.print("\n[bold blue]Fetching all checkpoints...[/bold blue]\n")

    # List all user checkpoints (fetch all pages)
    all_checkpoints = []
    offset = 0
    limit = 100

    while True:
        response = rest_client.list_user_checkpoints(
            limit=limit, offset=offset
        ).result()
        all_checkpoints.extend(response.checkpoints)

        # Check if there are more checkpoints to fetch
        if (
            response.cursor
            and response.cursor.offset + response.cursor.limit
            < response.cursor.total_count
        ):
            offset += limit
            console.print(
                f"[dim]Fetched {len(all_checkpoints)}/{response.cursor.total_count} checkpoints...[/dim]"
            )
        else:
            break

    console.print(f"[dim]Total checkpoints found: {len(all_checkpoints)}[/dim]\n")

    # Filter checkpoints containing filter_str in the tinker_path or checkpoint_id
    to_delete = []
    for ckpt in all_checkpoints:
        path = (ckpt.tinker_path or "").lower()
        checkpoint_id = (ckpt.checkpoint_id or "").lower()
        if filter_str in path or filter_str in checkpoint_id:
            to_delete.append(ckpt)

    if not to_delete:
        console.print(
            f"[yellow]No checkpoints found with '{args.filter}' in the name.[/yellow]"
        )
        return

    # Display checkpoints to be deleted
    table = Table(title=f"Found {len(to_delete)} checkpoint(s) matching '{args.filter}'")
    table.add_column("Checkpoint ID", style="cyan")
    table.add_column("Type", style="blue")
    table.add_column("Tinker Path", style="magenta")
    table.add_column("Size", style="green")
    table.add_column("Created", style="yellow")

    for ckpt in to_delete:
        size_mb = ckpt.size_bytes / (1024 * 1024) if ckpt.size_bytes else 0
        table.add_row(
            ckpt.checkpoint_id or "N/A",
            ckpt.checkpoint_type or "N/A",
            ckpt.tinker_path or "N/A",
            f"{size_mb:.2f} MB",
            ckpt.time.strftime("%Y-%m-%d %H:%M:%S") if ckpt.time else "Unknown",
        )

    console.print(table)
    console.print()

    # Confirm deletion
    response = console.input(
        "[bold red]Delete these checkpoints? This cannot be undone! (yes/no): [/bold red]"
    )

    if response.lower() not in ["yes", "y"]:
        console.print("[yellow]Deletion cancelled.[/yellow]")
        return

    # Delete checkpoints
    console.print("\n[bold red]Deleting checkpoints...[/bold red]\n")

    for i, ckpt in enumerate(to_delete, 1):
        try:
            console.print(
                f"[{i}/{len(to_delete)}] Deleting {ckpt.checkpoint_id} ({ckpt.checkpoint_type})...",
                end=" ",
            )
            rest_client.delete_checkpoint_from_tinker_path(ckpt.tinker_path).result()
            console.print("[green]✓ Done[/green]")
        except Exception as e:
            console.print(f"[red]✗ Failed: {e}[/red]")

    console.print("\n[bold green]Deletion complete![/bold green]\n")


if __name__ == "__main__":
    main()
