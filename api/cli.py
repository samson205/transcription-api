import argparse
import asyncio
import uuid
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

from api.core.dependencies import get_conversation_orchestrator
from api.core.warnings import configure_warnings
from api.core.config import settings

console = Console()
configure_warnings()


async def process_single_file(
    file_path: Path, output_dir: Path | None
) -> None:
    if file_path.suffix not in settings.EXTENSIONS:
        return
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
            output_dir.mkdir(parents=True, exist_ok=True)
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


async def process_directory(folder_path: Path, output_dir: Path | None):
    if not folder_path.exists() or not folder_path.is_dir():
        print(f"Ошибка: путь {folder_path} не существует или не является директорией")
        return

    if output_dir and not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    audio_files = [
        file
        for ext in settings.EXTENSIONS
        for file in folder_path.glob(f"*{ext.strip().lower()}")
    ]

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


async def main():
    parser = argparse.ArgumentParser(
        description="CLI утилита для пакетной обработки аудиозаписей"
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument("-f", "--file", type=Path, help="Файл для обработки (.mp3, .wav, .ogg)")
    group.add_argument("-d", "--directory", type=Path, help="Каталог с файлами для обработки")

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Путь к папке, куда сохранять JSON результаты (опционально)",
    )

    args = parser.parse_args()
    
    if args.directory:
        await process_directory(args.directory, args.output)
    elif args.file:
        await process_single_file(args.file, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
