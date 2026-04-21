#!/bin/bash
# =================================================================
# Feed Folder - Batch import files from a directory into nagual
# Supports: .md .txt .json .rs .toml .sql .pdf .png .jpg .jpeg .docx .html
# Usage: ./scripts/feed-folder.sh <folder> [domain] [db-path]
# =================================================================
set -e

FOLDER="${1:?Usage: feed-folder.sh <folder> [domain] [db-path]}"
DOMAIN="${2:-auto}"
DB_PATH="${3:-./nagual.db}"
NAGUAL="${NAGUAL_BIN:-nagual}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ ! -d "$FOLDER" ]; then
  echo "Error: $FOLDER is not a directory"
  exit 1
fi

echo "=== Nagual Feed: $FOLDER ==="
echo "Domain: $DOMAIN"
echo "Database: $DB_PATH"
echo ""

# Use Python for robust file content handling (no shell quoting issues)
python3 - "$FOLDER" "$DOMAIN" "$DB_PATH" "$NAGUAL" << 'PYEOF'
import sys, os, subprocess, json, glob, re, shutil, tempfile

folder = sys.argv[1]
domain = sys.argv[2]
db_path = sys.argv[3]
nagual = sys.argv[4]

# Check available tools
HAS_PDFTOTEXT = shutil.which('pdftotext') is not None
HAS_TESSERACT = shutil.which('tesseract') is not None
HAS_DOCX = False
try:
    from docx import Document as DocxDocument
    HAS_DOCX = True
except ImportError:
    pass

print(f"Tools: pdftotext={'yes' if HAS_PDFTOTEXT else 'NO'}, tesseract={'yes' if HAS_TESSERACT else 'NO'}, python-docx={'yes' if HAS_DOCX else 'NO'}")

# Find all importable files
EXTENSIONS = ('*.md', '*.txt', '*.json', '*.rs', '*.toml', '*.sql',
              '*.pdf', '*.png', '*.jpg', '*.jpeg', '*.docx', '*.html', '*.htm')
files = []
for ext in EXTENSIONS:
    files.extend(glob.glob(os.path.join(folder, '**', ext), recursive=True))
files.sort()

# Count by type
counts = {}
for f in files:
    e = os.path.splitext(f)[1][1:].lower()
    counts[e] = counts.get(e, 0) + 1
total = len(files)

count_str = ', '.join(f"{v} .{k}" for k, v in sorted(counts.items()))
print(f"Found: {count_str} ({total} total)")
print()

if total == 0:
    print("No importable files found.")
    sys.exit(0)

# Size limits by type (binary formats get more room)
SIZE_LIMITS = {
    'pdf': 50 * 1024 * 1024,  # 50MB for PDFs
    'docx': 2 * 1024 * 1024,  # 2MB for DOCX
    'png': 10 * 1024 * 1024,  # 10MB for images
    'jpg': 10 * 1024 * 1024,
    'jpeg': 10 * 1024 * 1024,
}
DEFAULT_SIZE_LIMIT = 51200  # 50KB for text files

def extract_pdf(filepath):
    """Extract text from PDF using pdftotext."""
    if not HAS_PDFTOTEXT:
        return None, None
    try:
        result = subprocess.run(
            ['pdftotext', '-layout', '-nopgbrk', filepath, '-'],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            return None, None
        text = result.stdout.strip()
        if not text:
            return None, None
        # Title: first non-empty line
        title = None
        for line in text.split('\n')[:20]:
            stripped = line.strip()
            if stripped and len(stripped) > 3:
                title = stripped[:200]
                break
        return title, text[:8000]
    except Exception:
        return None, None

def extract_image_ocr(filepath):
    """Extract text from image using tesseract OCR."""
    if not HAS_TESSERACT:
        return None, None
    try:
        result = subprocess.run(
            ['tesseract', filepath, 'stdout', '--psm', '3'],
            capture_output=True, text=True, timeout=60
        )
        text = result.stdout.strip()
        if not text or len(text) < 10:
            return None, None
        title = None
        for line in text.split('\n')[:10]:
            stripped = line.strip()
            if stripped and len(stripped) > 3:
                title = stripped[:200]
                break
        return title, text[:5000]
    except Exception:
        return None, None

def extract_docx(filepath):
    """Extract text from DOCX using python-docx."""
    if not HAS_DOCX:
        return None, None
    try:
        doc = DocxDocument(filepath)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        if not paragraphs:
            return None, None
        title = paragraphs[0][:200]
        text = '\n'.join(paragraphs)[:8000]
        return title, text
    except Exception:
        return None, None

def extract_html(filepath):
    """Extract text from HTML by stripping tags."""
    try:
        with open(filepath, 'r', errors='replace') as f:
            content = f.read()
        # Strip HTML tags
        text = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        if not text or len(text) < 10:
            return None, None
        # Title from <title> tag
        title_match = re.search(r'<title[^>]*>(.*?)</title>', content, re.IGNORECASE | re.DOTALL)
        title = title_match.group(1).strip()[:200] if title_match else None
        return title, text[:8000]
    except Exception:
        return None, None

imported = 0
skipped = 0
errors = 0

for filepath in files:
    rel_path = os.path.relpath(filepath, folder)
    ext = os.path.splitext(filepath)[1][1:].lower()
    size = os.path.getsize(filepath)

    # Size limit per type
    max_size = SIZE_LIMITS.get(ext, DEFAULT_SIZE_LIMIT)
    if size > max_size:
        print(f"  SKIP (>{max_size//1024}KB): {rel_path}")
        skipped += 1
        continue

    # Skip empty files
    if size == 0:
        print(f"  SKIP (empty): {rel_path}")
        skipped += 1
        continue

    # Auto-detect domain from folder structure
    dir_domain = os.path.dirname(rel_path).replace('/', '.').replace('\\', '.')
    if domain == "auto":
        file_domain = dir_domain if dir_domain and dir_domain != '.' else os.path.basename(folder)
    else:
        file_domain = domain

    title = ""
    solution = ""

    # ---- Binary formats (PDF, images, DOCX) ----
    if ext == 'pdf':
        title, solution = extract_pdf(filepath)
        if not solution:
            print(f"  SKIP (no text): {rel_path}")
            skipped += 1
            continue

    elif ext in ('png', 'jpg', 'jpeg'):
        title, solution = extract_image_ocr(filepath)
        if not solution:
            print(f"  SKIP (no OCR text): {rel_path}")
            skipped += 1
            continue

    elif ext == 'docx':
        title, solution = extract_docx(filepath)
        if not solution:
            print(f"  SKIP (no text): {rel_path}")
            skipped += 1
            continue

    elif ext in ('html', 'htm'):
        title, solution = extract_html(filepath)
        if not solution:
            print(f"  SKIP (no text): {rel_path}")
            skipped += 1
            continue

    # ---- Text formats ----
    else:
        try:
            with open(filepath, 'r', errors='replace') as f:
                content = f.read()
        except Exception as e:
            print(f"  SKIP (read error): {rel_path}: {e}")
            skipped += 1
            continue

        if ext == 'md':
            for line in content.split('\n')[:20]:
                if line.startswith('#'):
                    title = line.lstrip('#').strip()
                    break
            solution = content[:5000]
        elif ext == 'txt':
            lines = content.split('\n')
            title = lines[0].strip() if lines else os.path.basename(filepath)
            solution = '\n'.join(lines[1:])[:5000]
        elif ext == 'rs':
            doc_match = re.search(r'^//!\s*(.+)', content, re.MULTILINE)
            if doc_match:
                title = doc_match.group(1).strip()
            else:
                fn_match = re.search(r'pub\s+(fn|struct|enum|trait|mod)\s+(\w+)', content)
                if fn_match:
                    title = f"{fn_match.group(1)} {fn_match.group(2)} ({os.path.basename(filepath)})"
                else:
                    title = os.path.basename(filepath)
            solution = content[:5000]
        elif ext == 'toml':
            title = os.path.basename(filepath)
            solution = content[:5000]
        elif ext == 'sql':
            for line in content.split('\n')[:10]:
                stripped = line.strip()
                if stripped.startswith('--') and len(stripped) > 5:
                    title = stripped.lstrip('-').strip()
                    break
            if not title:
                title = os.path.basename(filepath)
            solution = content[:5000]
        elif ext == 'json':
            try:
                data = json.loads(content)
                if isinstance(data, dict):
                    title = data.get('problem', data.get('title', data.get('name', os.path.basename(filepath))))
                    solution = data.get('solution', data.get('content', data.get('description', content[:5000])))
                elif isinstance(data, list):
                    title = f"Collection from {os.path.basename(filepath)}"
                    solution = json.dumps(data[:10], indent=2)[:5000]
                else:
                    title = os.path.basename(filepath)
                    solution = content[:5000]
            except json.JSONDecodeError:
                print(f"  SKIP (invalid JSON): {rel_path}")
                skipped += 1
                continue

    if not title:
        title = os.path.basename(filepath)
    if not solution:
        solution = f"(imported from {rel_path})"

    tags = f"imported,{ext},{dir_domain.replace('.', ',')}" if dir_domain and dir_domain != '.' else f"imported,{ext}"

    clean_title = (title[:200] if title else os.path.basename(filepath)).replace('\x00', '')
    clean_solution = solution.replace('\x00', '')

    try:
        result = subprocess.run(
            [nagual, 'knowledge', 'store', clean_title,
             '--solution', clean_solution,
             '--domain', file_domain,
             '--tags', tags,
             '--db-path', db_path],
            capture_output=True, text=True, timeout=60
        )
        output = result.stdout + result.stderr
        if 'Knowledge Stored' in output:
            imported += 1
            print(f"  OK: {rel_path} -> {file_domain}")
        else:
            errors += 1
            print(f"  ERR: {rel_path}")
    except Exception as e:
        errors += 1
        print(f"  ERR: {rel_path}: {e}")

print()
print("=== Feed Complete ===")
print(f"  Imported: {imported}")
print(f"  Skipped: {skipped}")
print(f"  Errors: {errors}")
print(f"  Database: {db_path}")
PYEOF
