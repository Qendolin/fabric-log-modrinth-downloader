# AI Disclaimer

This project was made using Gemini 2.5 Pro. I just needed a script that can download mods from logs, and I wanted it done quickly.

# Minecraft Log Mod Downloader

This Python script automates the process of downloading mods for a Minecraft Fabric modpack directly from a `latest.log` file. It intelligently parses the log's mod list, finds the correct projects on Modrinth using fuzzy search, and selects the best matching version to download. On the "All of Fabric 7" modpack there is about a 90% success rate.

## Features

-   **Log Parsing**: Automatically finds and parses the mod list block within a Minecraft log file.
-   **Local Mod Index**: Builds a local index of all available Fabric mods for a specific Minecraft version. This is a one-time operation per version that makes future searches extremely fast and accurate.
-   **Intelligent Project Matching**: Uses a Levenshtein distance algorithm to "fuzzy match" mod IDs from the log to their official project names on Modrinth, correcting for common mismatches.
-   **Advanced Version Matching**: Employs a multi-tiered search logic to find the best version file, checking against Modrinth's version metadata and, crucially, the filenames themselves for the most reliable match.
-   **Persistent Caching**: Caches API results for mod versions in a local `modrinth_version_cache.json`, drastically speeding up subsequent runs by minimizing redundant API calls.
-   **Manual Overrides**: Allows you to create an `overrides.json` file to manually map difficult mod IDs to their correct Modrinth project "slug" or to intentionally skip them.
-   **Dry Run Mode**: Includes a `--dry-run` flag to perform a complete simulation of the process without downloading any files, allowing you to verify matches and versions first.
-   **Actionable Reporting**: Provides a clear summary upon completion, listing any mods it failed to find and any matches that were made with low confidence, so you know exactly what needs review.

## Requirements

-   Python 3.6+
-   The `requests` library
-   The `rapidfuzz` library

## How to Use

### 1. Installation

First, install the required Python library using pip:

```bash
pip install requests
```

### 2. (Optional) Create an Override File

For mods that are consistently mismatched or not available on Modrinth, you can create a file named `overrides.json` in the same directory as the script.

-   To map a mod ID to the correct Modrinth project, use the project's URL slug.
-   To intentionally skip a mod (e.g., it's a local-only library), set the value to an empty string (`""`).

**Example `overrides.json`:**

```json
{
  "yet_another_config_lib_v3": "yacl",
  "a-mod-with-a-weird-name": "the-correct-modrinth-slug",
  "a_local_mod_not_on_modrinth": ""
}
```

### 3. Run the Script

Execute the script from your terminal, providing the path to your Minecraft log file. Mods will be downloaded into a `downloaded_mods` folder.

**Standard Run:**

```bash
python download_mods.py /path/to/your/minecraft/logs/latest.log
```

**Dry Run (to check matches without downloading):**

```bash
python download_mods.py /path/to/your/minecraft/logs/latest.log --dry-run
```

The script will first build the necessary indexes (if they don't exist) and then proceed to find and download the mods. Upon completion, it will print a summary of its actions.