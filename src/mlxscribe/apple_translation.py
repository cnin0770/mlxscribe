"""On-device translation through Apple's Translation framework.

mlxscribe talks to the framework through a small Swift helper. The helper's source lives
in this module; it is compiled with ``swiftc`` on first use and cached by source hash, so
editing the source rebuilds it and nothing has to be installed beyond Xcode's command-line
tools. Language packs are managed by macOS, in System Settings.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

ENGINE_NAME = "apple"
INSTALL_HINT = (
    "Install it in System Settings > General > Language & Region > Translation Languages, "
    "then run the command again."
)

# Whisper reports bare codes; Apple's Translation framework wants a script for Chinese.
SOURCE_LANGUAGE_ALIASES = {"zh": "zh-Hans"}

HELPER_SOURCE = r"""
import Foundation
import Translation

struct Item: Codable {
    let id: Int
    let text: String
}

func fail(_ message: String) -> Never {
    FileHandle.standardError.write((message + "\n").data(using: .utf8)!)
    exit(1)
}

func statusName(_ status: LanguageAvailability.Status) -> String {
    switch status {
    case .installed: return "installed"
    case .supported: return "supported"
    case .unsupported: return "unsupported"
    @unknown default: return "unknown"
    }
}

@main
struct MlxscribeAppleTranslate {
    static func main() async {
        let args = CommandLine.arguments
        guard args.count == 4, ["check", "translate"].contains(args[1]) else {
            fail("usage: mlxscribe-apple-translate check|translate <source> <target>")
        }
        let source = Locale.Language(identifier: args[2])
        let target = Locale.Language(identifier: args[3])

        if args[1] == "check" {
            let status = await LanguageAvailability().status(from: source, to: target)
            print(statusName(status))
            return
        }

        let input = FileHandle.standardInput.readDataToEndOfFile()
        let items: [Item]
        do {
            items = try JSONDecoder().decode([Item].self, from: input)
        } catch {
            fail("could not read segments from stdin: \(error)")
        }
        let session = TranslationSession(installedSource: source, target: target)
        let requests = items.map {
            TranslationSession.Request(sourceText: $0.text, clientIdentifier: String($0.id))
        }
        do {
            let responses = try await session.translations(from: requests)
            let output = responses.map {
                Item(id: Int($0.clientIdentifier ?? "") ?? -1, text: $0.targetText)
            }
            FileHandle.standardOutput.write(try JSONEncoder().encode(output))
        } catch {
            fail("translation failed: \(error)")
        }
    }
}
"""


def cache_dir() -> Path:
    base = os.environ.get("MLXSCRIBE_CACHE_DIR")
    if base:
        return Path(base).expanduser()
    return Path.home() / "Library" / "Caches" / "mlxscribe"


def helper_path() -> Path:
    digest = hashlib.sha256(HELPER_SOURCE.encode("utf-8")).hexdigest()[:12]
    return cache_dir() / f"apple-translate-{digest}"


def ensure_helper() -> Path:
    """Return the compiled helper, building it first if this source has not been built yet."""
    path = helper_path()
    if path.exists():
        return path
    if platform.system() != "Darwin":
        raise RuntimeError("--engine apple needs macOS")
    swiftc = shutil.which("swiftc")
    if not swiftc:
        raise RuntimeError(
            "--engine apple needs swiftc to build its helper; install Xcode's command-line tools "
            "with `xcode-select --install`"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    source = path.with_suffix(".swift")
    source.write_text(HELPER_SOURCE, encoding="utf-8")
    building = path.with_name(path.name + ".building")
    print("Building the Apple Translation helper (first use only)...")
    completed = subprocess.run(
        [swiftc, "-parse-as-library", "-O", str(source), "-o", str(building)],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "could not build the Apple Translation helper; this macOS may be too old for "
            f"command-line translation.\n{completed.stderr.strip()}"
        )
    building.replace(path)
    return path


def apple_language(code: str) -> str:
    return SOURCE_LANGUAGE_ALIASES.get(code, code)


def check_language_pair(source_language: str, target_language: str, helper: Path | None = None) -> str:
    """Return the pair's status (installed, supported, unsupported), or raise if unusable."""
    helper = helper or ensure_helper()
    source, target = apple_language(source_language), apple_language(target_language)
    completed = subprocess.run(
        [str(helper), "check", source, target], capture_output=True, text=True
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Apple Translation check failed: {completed.stderr.strip()}")
    status = completed.stdout.strip()
    if status == "unsupported":
        raise RuntimeError(f"Apple Translation does not support {source} -> {target}")
    if status != "installed":
        raise RuntimeError(f"The {source} -> {target} language pack is not installed. {INSTALL_HINT}")
    return status


def macos_version() -> str:
    return platform.mac_ver()[0] or "unknown"


class AppleTranslator:
    """Translate batches of ``{"id", "text"}`` items with the compiled helper."""

    def __init__(self, source_language: str, target_language: str, helper: Path | None = None):
        self.helper = helper or ensure_helper()
        self.source = apple_language(source_language)
        self.target = apple_language(target_language)

    def __call__(self, batch: list[dict]) -> tuple[list[dict], dict]:
        # The framework rejects empty strings; blank segments stay blank.
        items = [
            {"id": int(segment["id"]), "text": str(segment["text"]).strip()}
            for segment in batch
            if str(segment["text"]).strip()
        ]
        translations = []
        if items:
            completed = subprocess.run(
                [str(self.helper), "translate", self.source, self.target],
                input=json.dumps(items, ensure_ascii=False),
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                message = completed.stderr.strip() or f"exit code {completed.returncode}"
                raise RuntimeError(f"Apple Translation failed: {message}")
            translations = [
                {"id": item["id"], "text": str(item["text"]).strip()}
                for item in json.loads(completed.stdout)
            ]
        return translations, {"batches": 1}
