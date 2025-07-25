import re
import os
import sys
import json
import time
import requests
from urllib.parse import quote
from rapidfuzz import process, fuzz

# --- Configuration ---
# Script settings that can be adjusted by the user.
MODRINTH_API_BASE = "https://api.modrinth.com/v2"
MINECRAFT_VERSION = None  # Auto-detected from log if None
LOADER = "fabric"
DOWNLOAD_FOLDER = "downloaded_mods"
USER_AGENT = "Qendolin/fabric-log-modrinth-downloader"

# Thresholds for matching logic.
# Minimum score (0.0-1.0) to consider a fuzzy match valid.
FUZZY_SEARCH_THRESHOLD = 0.6
# Matches with scores below this are deferred for user review.
LOW_CONFIDENCE_REPORTING_THRESHOLD = 0.9
# A fixed high score for acronym matches.
ACRONYM_CONFIDENCE_SCORE = 0.95

# Local file names for caching and overrides.
VERSION_CACHE_FILE = "modrinth_version_cache.json"
OVERRIDES_FILE = "overrides.json"


# --- Caches (Loaded from files) ---
VERSION_CACHE = {}
OVERRIDES = {}


# --- API & Rate Limiting ---


class RateLimiter:
    """A simple client-side rate limiter to avoid spamming the Modrinth API."""

    def __init__(self, requests_per_minute):
        self.max_requests = requests_per_minute
        self.window_seconds = 60
        self.request_count = 0
        self.window_start = time.monotonic()

    def wait(self):
        """Checks the rate limit and waits if necessary before allowing a request."""
        elapsed = time.monotonic() - self.window_start
        if elapsed > self.window_seconds:
            self.request_count = 0
            self.window_start = time.monotonic()

        if self.request_count >= self.max_requests:
            time_to_wait = self.window_seconds - (time.monotonic() - self.window_start)
            if time_to_wait > 0:
                print(f"Rate limit reached. Waiting for {time_to_wait:.2f} seconds...")
                time.sleep(time_to_wait)
            self.request_count = 0
            self.window_start = time.monotonic()
        self.request_count += 1


# Global instance of the rate limiter. Using 290 to be safe against Modrinth's 300/min limit.
MODRINTH_RATELIMITER = RateLimiter(requests_per_minute=290)


def rate_limited_get(*args, **kwargs):
    """A wrapper for requests.get() that respects our rate limit."""
    MODRINTH_RATELIMITER.wait()
    return requests.get(*args, **kwargs)


# --- String & Version Utilities ---


def normalize_string(s: str) -> str:
    """Converts a string to lowercase and removes all non-alphanumeric characters."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def normalize_for_matching(s: str) -> str:
    """
    Performs advanced normalization on a project title for better matching by
    first removing content in parentheses or brackets.
    """
    s_cleaned = re.sub(r"[\(\[].*?[\)\]]", "", s)
    return normalize_string(s_cleaned)


def generate_acronym(s: str) -> str:
    """Generates a simple acronym from a string based on word boundaries."""
    parts = [part for part in re.split(r"[\s_-]+", s) if part]
    return "".join(part[0] for part in parts).lower()


def extract_version_string(version: str) -> str | None:
    """Extracts the core version number (e.g., '1.2.3') from a complex filename."""
    match = re.search(r"\d+(\.\d+)*([.+-]?[a-zA-Z0-9_.-]+)*", version)
    return match.group(0) if match else None


# --- Cache & Override Handling ---


def load_caches_and_overrides():
    """Loads the version cache and manual overrides from their respective JSON files."""
    global VERSION_CACHE, OVERRIDES
    if os.path.exists(VERSION_CACHE_FILE):
        with open(VERSION_CACHE_FILE, "r", encoding="utf-8") as f:
            try:
                VERSION_CACHE = json.load(f)
            except json.JSONDecodeError:
                VERSION_CACHE = {}
    if os.path.exists(OVERRIDES_FILE):
        with open(OVERRIDES_FILE, "r", encoding="utf-8") as f:
            try:
                OVERRIDES = json.load(f)
            except json.JSONDecodeError:
                print(f"  WARN: Could not read '{OVERRIDES_FILE}'.")


def save_version_cache():
    """Saves the in-memory version cache to a JSON file to speed up future runs."""
    print(f"\nSaving version cache to '{VERSION_CACHE_FILE}'...")
    with open(VERSION_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(VERSION_CACHE, f, indent=2)


# --- Core Modrinth Logic ---


def build_or_load_index(mc_version: str) -> dict:
    """
    Builds a local index of all mods for a specific Minecraft version.
    This is a one-time, expensive operation per version. The resulting JSON
    is saved to disk to make subsequent runs instantaneous.
    """
    index_filename = f"modrinth_{LOADER}_{mc_version}_index.json"
    if os.path.exists(index_filename):
        print(f"Loading mod index from '{index_filename}'...")
        with open(index_filename, "r", encoding="utf-8") as f:
            return json.load(f)

    print(f"Building new index for MC {mc_version}...")
    index, offset, total_hits = {}, 0, -1
    while total_hits == -1 or offset < total_hits:
        facets = f'[["project_type:mod"], ["categories:{LOADER}"], ["versions:{mc_version}"]]'
        search_url = f"{MODRINTH_API_BASE}/search?limit=100&offset={offset}&facets={quote(facets)}"
        try:
            response = rate_limited_get(search_url, headers={"User-Agent": USER_AGENT})
            response.raise_for_status()
            data = response.json()
            if not data["hits"]:
                break

            for hit in data["hits"]:
                # Pre-calculate normalized forms and acronyms for faster searching later.
                index[hit["slug"]] = {
                    "title": hit["title"],
                    "normalized_title": normalize_for_matching(hit["title"]),
                    "title_acronym": generate_acronym(hit["title"]),
                    "normalized_slug": normalize_string(hit["slug"]),
                    "slug_acronym": generate_acronym(hit["slug"]),
                }

            if total_hits == -1:
                total_hits = data["total_hits"]
            offset += len(data["hits"])
            print(f"  Fetched {offset}/{total_hits} projects...")
        except requests.RequestException as e:
            print(f"ERROR: Failed to fetch API data: {e}")
            return {}

    with open(index_filename, "w", encoding="utf-8") as f:
        json.dump(index, f)
    print(f"Successfully saved index to '{index_filename}'.")
    return index


def find_mod_in_index(mod_name: str, index: dict) -> tuple[str, float] | None:
    """
    Finds the best Modrinth project slug for a given mod name using a tiered approach:
    1. Exact Match: Checks for a direct match of the mod ID to a Modrinth slug.
    2. Acronym Match: Checks if the mod ID is an acronym for a project title/slug or vice-versa.
    3. Fuzzy Match: Uses high-performance fuzzy string matching on titles and slugs.
    """
    # Tier 1: Exact slug match (highest confidence).
    if mod_name in index:
        return mod_name, 1.0

    normalized_mod_name = normalize_string(mod_name)
    mod_name_acronym = generate_acronym(mod_name)

    # Tier 2: Acronym Heuristic.
    if len(mod_name_acronym) > 3:
        for slug, data in index.items():
            if (
                mod_name_acronym == data["title_acronym"]
                or mod_name_acronym == data["slug_acronym"]
            ):
                print(f"  Acronym match: '{data['title']}' (Heuristic).")
                return slug, ACRONYM_CONFIDENCE_SCORE
    for slug, data in index.items():
        if (
            len(data["title_acronym"]) > 3
            and data["title_acronym"] == normalized_mod_name
        ):
            print(f"  Acronym match: '{data['title']}' (Heuristic).")
            return slug, ACRONYM_CONFIDENCE_SCORE

    # Tier 3: High-performance fuzzy search using the rapidfuzz library.
    slugs = list(index.keys())
    titles_to_search = [data["normalized_title"] for data in index.values()]
    slugs_to_search = [data["normalized_slug"] for data in index.values()]

    # Use a score cutoff to tell rapidfuzz to ignore any matches below our threshold.
    # The library uses a 0-100 scale, so we multiply our 0.0-1.0 threshold by 100.
    score_cutoff = FUZZY_SEARCH_THRESHOLD * 100

    # Find the best match against both titles and slugs.
    title_match = process.extractOne(
        normalized_mod_name,
        titles_to_search,
        scorer=fuzz.ratio,
        score_cutoff=score_cutoff,
    )
    slug_match = process.extractOne(
        normalized_mod_name,
        slugs_to_search,
        scorer=fuzz.ratio,
        score_cutoff=score_cutoff,
    )

    # Compare the results from title and slug matching and select the one with the higher score.
    best_match_slug, highest_score = None, -1.0
    if title_match and slug_match:
        if title_match[1] >= slug_match[1]:
            highest_score = title_match[1] / 100.0
            best_match_slug = slugs[title_match[2]]
        else:
            highest_score = slug_match[1] / 100.0
            best_match_slug = slugs[slug_match[2]]
    elif title_match:
        highest_score = title_match[1] / 100.0
        best_match_slug = slugs[title_match[2]]
    elif slug_match:
        highest_score = slug_match[1] / 100.0
        best_match_slug = slugs[slug_match[2]]

    if best_match_slug:
        print(
            f"  Fuzzy match: '{index[best_match_slug]['title']}' (Score: {highest_score:.2f})."
        )
        return best_match_slug, highest_score
    else:
        print(f"  No confident match found for '{mod_name}'.")
        return None


def get_project_versions(project_slug: str) -> list | None:
    """Retrieves all available versions for a given project slug, caching the results."""
    cache_key = f"{project_slug}-{MINECRAFT_VERSION}"
    if cache_key in VERSION_CACHE:
        return VERSION_CACHE[cache_key]

    print(f"  Querying API for versions of '{project_slug}'...")
    versions_url = (
        f"{MODRINTH_API_BASE}/project/{project_slug}/version?"
        f'loaders=["{LOADER}"]&game_versions=["{MINECRAFT_VERSION}"]'
    )
    try:
        response = rate_limited_get(versions_url, headers={"User-Agent": USER_AGENT})
        response.raise_for_status()
        versions = response.json()
        if versions:
            VERSION_CACHE[cache_key] = versions
        return versions
    except requests.RequestException as e:
        print(f"  ERROR: Could not fetch versions for '{project_slug}': {e}")
        return None


def find_best_version(log_version: str, all_versions: list) -> dict | None:
    """
    Finds the best version object from a list using a multi-tiered search,
    from exact matches down to comparing core version numbers.
    """
    for v in all_versions:
        if v["version_number"] == log_version:
            print(f"  Found exact version_number match: '{log_version}'.")
            return v
        if v.get("files") and log_version in v["files"][0]["filename"]:
            print(f"  Found exact string in filename: '{v['files'][0]['filename']}'.")
            return v

    log_core = extract_version_string(log_version)
    if not log_core:
        return all_versions[0]  # Fallback if we can't parse the log version.

    for v in all_versions:
        if extract_version_string(v["version_number"]) == log_core:
            print(
                f"  Found core version match in version_number: '{v['version_number']}'."
            )
            return v
        if (
            v.get("files")
            and extract_version_string(v["files"][0]["filename"]) == log_core
        ):
            print(
                f"  Found core version match in filename: '{v['files'][0]['filename']}'."
            )
            return v
        remote_core = extract_version_string(v["version_number"])
        if remote_core and log_core in remote_core:
            print(
                f"  Found substring match in version_number: '{v['version_number']}'."
            )
            return v

    # If all else fails, fall back to the newest compatible version.
    print(
        f"  WARN: No strong version match for '{log_version}'. Using newest compatible."
    )
    return all_versions[0]


def download_mod_file(version_data: dict, dry_run: bool = False):
    """Downloads the primary file from a version object into the download folder."""
    if not version_data or not version_data.get("files"):
        print("  ERROR: Invalid version data provided for download.")
        return False

    file_info = version_data["files"][0]
    file_name = file_info["filename"]

    if dry_run:
        print(f"  DRY RUN: Would download '{file_name}'.")
        return True

    file_path = os.path.join(DOWNLOAD_FOLDER, file_name)
    if os.path.exists(file_path):
        print(f"  Skipping '{file_name}', already exists.")
        return True

    print(f"  Downloading '{file_name}'...")
    try:
        with rate_limited_get(
            file_info["url"], stream=True, headers={"User-Agent": USER_AGENT}
        ) as r:
            r.raise_for_status()
            with open(file_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"  Successfully downloaded '{file_name}'.")
        return True
    except requests.RequestException as e:
        print(f"  ERROR: Failed to download '{file_name}': {e}")
        return False


# --- Main Execution & User Interaction ---


def parse_log_for_mods(log_content: str) -> tuple[list, str | None]:
    """Parses a Minecraft log file to extract the list of mods and the game version."""
    start_match = re.search(
        r"^\[.+?\] \[.+?\/.+?\]: Loading \d+ mods:$", log_content, re.MULTILINE
    )
    if not start_match:
        return [], None

    content_after = log_content[start_match.end() :].lstrip()
    end_match = re.search(r"^\s*\[", content_after, re.MULTILINE)
    mod_block = content_after[: end_match.start()] if end_match else content_after

    mods, mc_version = [], None
    mc_match = re.search(r"^\t- minecraft (.+)$", mod_block, re.MULTILINE)
    if mc_match:
        mc_version = mc_match.group(1).strip()

    # Iterate over the mod list block and extract mod IDs and versions.
    for match in re.finditer(r"^\t- ([\w.-]+) (.+)$", mod_block, re.MULTILINE):
        # Ignore non-mod entries like minecraft, java, and the loader itself.
        if match.group(1).lower() not in ["minecraft", "java", "fabricloader"]:
            mods.append((match.group(1).strip(), match.group(2).strip()))
    return mods, mc_version


def handle_deferred_downloads(deferred_list: list, dry_run: bool) -> tuple[int, list]:
    """
    Prompts the user to decide the fate of low-confidence matches.
    Returns the number of successful downloads and a list of mods the user skipped.
    """
    if not deferred_list:
        return 0, []

    print("\n--- Low Confidence Matches Require Review ---")
    for i, (name, title, slug, score, _, mod_version) in enumerate(deferred_list):
        print(
            f"  {i+1}. Mod ID: '{name}' v'{mod_version}' -> Match: '{title}' (slug: {slug}) (Score: {score:.2f})"
        )

    while True:
        choice = input(
            "\nChoose an action: [d]ownload all, decide [i]ndividually, [s]kip all? "
        ).lower()
        if choice in ["d", "i", "s"]:
            break
        print("Invalid choice. Please enter 'd', 'i', or 's'.")

    success_count, failures = 0, []

    if choice == "d":
        print("Downloading all suggested mods...")
        for _, _, _, _, version_info, _ in deferred_list:
            if download_mod_file(version_info, dry_run=dry_run):
                success_count += 1
    elif choice == "i":
        for i, (name, title, slug, _, version_info, mod_version) in enumerate(
            deferred_list
        ):
            while True:
                prompt = f"  Download '{title}' (slug: {slug}) for mod id '{name}' v'{mod_version}'? [y/n] "
                sub_choice = input(prompt).lower()
                if sub_choice in ["y", "n"]:
                    break
            if sub_choice == "y":
                if download_mod_file(version_info, dry_run=dry_run):
                    success_count += 1
            else:
                print(f"  Skipping '{title}'.")
                failures.append((name, "Skipped by user"))
    else:  # choice == 's'
        print("Skipping all low confidence mods.")
        for name, _, _, _, _, _ in deferred_list:
            failures.append((name, "Skipped by user"))

    return success_count, failures


def main():
    """Main execution function for the script."""
    dry_run = "--dry-run" in sys.argv
    if dry_run:
        sys.argv.remove("--dry-run")
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <path_to_log> [--dry-run]")
        sys.exit(1)
    if dry_run:
        print("--- DRY RUN MODE ENABLED ---\n")

    log_file_path = sys.argv[1]
    if not os.path.exists(log_file_path):
        print(f"Error: Log not found '{log_file_path}'")
        sys.exit(1)
    if not dry_run and not os.path.exists(DOWNLOAD_FOLDER):
        os.makedirs(DOWNLOAD_FOLDER)

    load_caches_and_overrides()

    with open(log_file_path, "r", encoding="utf-8") as f:
        log_content = f.read()
    mods, detected_mc_version = parse_log_for_mods(log_content)

    global MINECRAFT_VERSION
    if not MINECRAFT_VERSION:
        MINECRAFT_VERSION = detected_mc_version
    if not MINECRAFT_VERSION:
        print("Error: Minecraft version not detected in log.")
        sys.exit(1)

    print(f"Detected Minecraft {MINECRAFT_VERSION} for {LOADER}.")
    mod_index = build_or_load_index(MINECRAFT_VERSION)
    if not mod_index:
        print("Could not build or load the mod index. Exiting.")
        sys.exit(1)
    if not mods:
        print("No mods found in log file.")
        return

    print(f"\nFound {len(mods)} mods to process...\n")
    success_count = 0
    unmatched = []
    deferred_downloads = []
    try:
        for mod_name, mod_version in mods:
            print(f"Processing '{mod_name}' v'{mod_version}'...")

            # Step 1: Check for manual overrides.
            if mod_name in OVERRIDES:
                override_slug = OVERRIDES[mod_name]
                if override_slug == "":
                    print(f"  Override rule: Skipping '{mod_name}' intentionally.")
                    unmatched.append((mod_name, "Skipped by override rule"))
                    continue
                else:
                    print(f"  Override rule: Using slug '{override_slug}'.")
                    project_slug, score = override_slug, 1.0
            else:
                # Step 2: Find the project on Modrinth using our tiered matching.
                match_result = find_mod_in_index(mod_name, mod_index)
                if not match_result:
                    unmatched.append((mod_name, "Project not found on Modrinth"))
                    continue
                project_slug, score = match_result

            # Step 3: Get all available versions for the matched project.
            all_versions = get_project_versions(project_slug)
            if not all_versions:
                unmatched.append((mod_name, "No compatible version found"))
                continue

            # Step 4: Find the best specific version file.
            version_info = find_best_version(mod_version, all_versions)

            # Step 5: Based on confidence score, either download immediately or defer.
            if score < LOW_CONFIDENCE_REPORTING_THRESHOLD:
                print(f"  Match confidence is low ({score:.2f}). Deferring decision.")
                deferred_downloads.append(
                    (
                        mod_name,
                        mod_index[project_slug]["title"],
                        project_slug,
                        score,
                        version_info,
                        mod_version,
                    )
                )
            else:
                if download_mod_file(version_info, dry_run=dry_run):
                    success_count += 1

    finally:
        # Ensure the cache is always saved, even if the script is interrupted.
        save_version_cache()

    # After processing all mods, handle the ones that were deferred.
    if deferred_downloads:
        deferred_success, deferred_failures = handle_deferred_downloads(
            deferred_downloads, dry_run
        )
        success_count += deferred_success
        unmatched.extend(deferred_failures)

    # --- Final Report ---
    print("\n" + "=" * 15 + " Run Summary " + "=" * 15)
    print(f"Successful: {success_count}\nFailed/Skipped: {len(unmatched)}")
    if not dry_run:
        print(f"Mods saved to: '{DOWNLOAD_FOLDER}/'")

    if unmatched:
        print("\n--- Unmatched or Skipped Mods ---")
        for name, reason in sorted(unmatched):
            print(f"- {name}: {reason}")

    print("\n" + "=" * (15 * 2 + 13))


if __name__ == "__main__":
    main()
