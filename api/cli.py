import argparse
import asyncio
import uuid
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

from api.core.dependencies import get_conversation_orchestrator
from api.core.warnings import configure_warnings

console = Console()
configure_warnings()


async def process_single_file(
    file_path: Path, output_dir: Path | None
) -> None:
    orchestrator = get_conversation_orchestrator()
    generated_id = uuid.uuid4()

    try:
        with console.status(
            f"[cyan]Обработка[/cyan] {file_path.name}..."
        ):
            result = await orchestrator.process_and_get_conversation(
                generated_id, file_path.name, str(file_path)
            )

        console.print(
            f"✅ [green]Успешно:[/green] {file_path.name} "
            f"[dim](ID: {generated_id})[/dim]"
        )

        if output_dir:
            output_file = output_dir / f"{file_path.stem}_{generated_id}.json"
            output_file.write_text(
                result.model_dump_json(indent=4),
                encoding="utf-8",
            )

            console.print(
                f"JSON сохранён: [blue]{output_file}[/blue]\n"
            )

    except Exception as e:
        console.print(
            f"❌ [red]Ошибка:[/red] {file_path.name}\n"
            f"[dim]{e}[/dim]\n"
        )


async def main():
    parser = argparse.ArgumentParser(
        description="CLI утилита для пакетной обработки аудиозаписей"
    )
    parser.add_argument("folder_path", type=str, help="Путь к папке с аудиофайлами")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Путь к папке, куда созранять JSON результаты (опционально)",
    )

    args = parser.parse_args()

    folder = Path(args.folder_path)
    output_dir = Path(args.output) if args.output else None
    if not folder.exists() or not folder.is_dir():
        print(f"Ошибка: путь {folder} не существует или не является директорией")
        return

    if output_dir and not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    extensions = ("*.mp3", "*.wav", "*.ogg")
    audio_files = []
    for ext in extensions:
        audio_files.extend(folder.glob(ext))

    if not audio_files:
        print("Аудиофайлы для обработки не найдены")
        return

    console.print(
        Panel.fit(
            f"Найдено файлов для обработки: [bold cyan]{len(audio_files)}[/bold cyan]",
            title="Transcription CLI",
        )
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:

        task = progress.add_task(
            "Обработка файлов...",
            total=len(audio_files),
        )

        for file_path in audio_files:
            await process_single_file(file_path, output_dir)
            progress.advance(task)


if __name__ == "__main__":
    asyncio.run(main())
