# nagual-wasm

WASM runtime for ProfDAG pattern search in browser/edge environments.

## Features

- **In-memory pattern index** with cosine similarity search
- **IndexedDB persistence** for patterns
- **JSON import/export** for pattern data
- **Vector similarity search** optimized for browser (< 10ms for 10K patterns)
- **Small bundle size** (< 2MB target)

## Building

### Prerequisites

```bash
# Install wasm-pack
cargo install wasm-pack

# Or with curl
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
```

### Build for Browser

```bash
# Development build
wasm-pack build --target web --dev

# Production build (optimized)
wasm-pack build --target web --release

# For bundlers (webpack, rollup, etc.)
wasm-pack build --target bundler --release
```

### Build Output

The build produces a `pkg/` directory with:
- `nagual_wasm_bg.wasm` - The WASM binary
- `nagual_wasm.js` - JavaScript bindings
- `nagual_wasm.d.ts` - TypeScript definitions
- `package.json` - npm package manifest

## Usage

### Basic Example

```javascript
import init, { WasmProfDAG, SearchConfig, generate_uuid, generate_random_embedding } from './pkg/nagual_wasm.js';

async function main() {
    // Initialize WASM
    await init();

    // Create ProfDAG instance
    const profdag = new WasmProfDAG();

    // Add patterns
    const embedding = generate_random_embedding(128);
    profdag.add_pattern('pattern-1', 'How to handle database timeouts', embedding);

    // Search for similar patterns
    const query = generate_random_embedding(128);
    const results = profdag.search(query, 5);

    console.log('Search results:', results);

    // Export/Import JSON
    const json = profdag.export_json();
    profdag.import_json(json);

    // IndexedDB persistence
    await profdag.save_to_indexeddb();
    await profdag.load_from_indexeddb();
}

main();
```

### Custom Configuration

```javascript
const config = new SearchConfig()
    .with_embedding_dim(128)
    .with_min_similarity(0.5)
    .with_max_results(20);

const profdag = WasmProfDAG.with_config(config);
```

### Pattern Types

```javascript
// Add pattern with full metadata
profdag.add_pattern_full({
    id: 'pattern-1',
    content: 'Pattern description',
    embedding: Array.from(new Float32Array(128)),
    pattern_type: 'pattern', // 'pattern', 'trajectory', 'prediction', 'decision'
    confidence: 0.8,
    metadata: { tags: ['important'] },
    created_at: new Date().toISOString()
});
```

## API Reference

### WasmProfDAG

Main class for pattern storage and search.

| Method | Description |
|--------|-------------|
| `new()` | Create with default config |
| `with_config(config)` | Create with custom config |
| `add_pattern(id, content, embedding)` | Add a pattern |
| `add_pattern_full(pattern)` | Add pattern with metadata |
| `remove_pattern(id)` | Remove by ID |
| `get_pattern(id)` | Get by ID |
| `search(embedding, top_k)` | Find similar patterns |
| `batch_search(embeddings, top_k)` | Batch similarity search |
| `pattern_count()` | Number of patterns |
| `is_empty()` | Check if empty |
| `clear()` | Remove all patterns |
| `export_json()` | Export as JSON |
| `import_json(json)` | Import from JSON |
| `save_to_indexeddb()` | Save to browser storage |
| `load_from_indexeddb()` | Load from browser storage |
| `clear_indexeddb()` | Clear browser storage |
| `get_stats()` | Get index statistics |

### SearchConfig

Configuration for search behavior.

| Property | Default | Description |
|----------|---------|-------------|
| `embedding_dim` | 128 | Expected embedding dimension |
| `min_similarity` | 0.0 | Minimum similarity threshold |
| `max_results` | 50 | Maximum results per search |

### Utilities

| Function | Description |
|----------|-------------|
| `generate_uuid()` | Generate random UUID |
| `generate_random_embedding(dim)` | Generate random normalized embedding |
| `version()` | Get module version |
| `build_info()` | Get build information |

## Performance

### Targets

- Search latency: < 10ms for 10K patterns
- Bundle size: < 2MB (WASM + JS)
- Memory: ~100 bytes per pattern (excluding embeddings)

### Benchmarks

Run benchmarks with:

```bash
wasm-pack test --headless --chrome
```

## Demo

Open `www/index.html` in a browser to see the demo page.

Features:
- Add patterns manually or generate random ones
- Search with text queries (mock embeddings)
- Import/export JSON
- IndexedDB persistence
- Performance timing

## Development

### Testing

```bash
# Unit tests (native)
cargo test

# WASM tests
wasm-pack test --headless --chrome
wasm-pack test --headless --firefox
```

### Linting

```bash
cargo clippy --target wasm32-unknown-unknown
```

## Bundle Size Optimization

The crate is optimized for small bundle size:

- `opt-level = "s"` for size optimization
- LTO (Link Time Optimization) enabled
- Single codegen unit for better optimization
- No debug info in release builds

To further reduce size:

```bash
# Use wasm-opt (requires binaryen)
wasm-opt -Oz pkg/nagual_wasm_bg.wasm -o pkg/nagual_wasm_bg.wasm
```

## License

MIT
