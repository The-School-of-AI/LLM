# Managing Version Complexity in UV Workspaces

When experimenting with different versions of libraries (e.g., `v4` vs `v5`), you can use two primary strategies to manage dependencies within a monorepo.

---

## Option 1: Dependency Groups (Soft Isolation)

**Best for**: When experiments can share a core environment but need optional "extra" libraries that are version-compatible.

### Option 1 Implementation

Add specific groups to your team's `pyproject.toml`:

```toml
[project]
name = "your-project-name"
dependencies = [
    "numpy>=2.4.2",  # Shared core dependencies
]

[dependency-groups]
v4-engine = [
    "scikit-learn==1.2.0", 
]
v5-engine = [
    "ray>=2.53.0",         
    "faiss-cpu>=1.13.2",
]
```

### Option 1 Usage

* **Work on V4**: `uv sync --group v4-engine`
* **Work on V5**: `uv sync --group v5-engine`

> [!IMPORTANT]
> **Admin Considerations (Option 1)**:
> Since dependency groups are optional, a standard root `uv sync` **will not** check them.
> Admins MUST run:
>
> ```bash
> uv sync --all-groups
> ```
>
> This ensures that experimental groups don't "drift" or become unresolvable over time.

> [!WARNING]
> **Limitation**: Since there is only one root `uv.lock`, you **cannot** have conflicting versions of the same package (e.g., trying to use `pydantic v1` in the V4 group and `pydantic v2` in the V5 group).

---

## Option 2: Workspace Members (Hard Isolation)

**Best for**: When experiments have breaking version conflicts (e.g., different `torch` or `pydantic` requirements) or need completely independent environments.

### Option 2 Implementation

Treat each experiment as a separate "package" within the workspace.

**Directory Structure:**

```text
LLM/
├── pyproject.toml (Root Workspace)
├── experiments/
│   └── your_team_folder/
│       ├── v4/
│       │   └── pyproject.toml
│       └── v5/
│           └── pyproject.toml
```

**Root `pyproject.toml`:**

```toml
[tool.uv]
workspace = { members = [
    "experiments/your_team_folder/v4",
    "experiments/your_team_folder/v5"
] }
```

**Member `pyproject.toml` (e.g., V4):**

```toml
[project]
name = "engine-v4"
dependencies = [ "pydantic<2.0.0" ]
```

### Option 2 Usage

Navigate to the specific folder and create a dedicated environment.

* **Work on V4**:

    ```bash
    cd experiments/your_team_folder/v4
    uv venv .venv
    uv sync
    ```

> [!IMPORTANT]
> **Admin Considerations (Option 2)**:
> This is the "Zero-Maintenance" option for Admins. A standard global `uv sync` automatically discovers all workspace members and validates their dependencies. No special flags are required.

---

## Comparison Summary

| Feature | Option 1: Groups (Soft) | Option 2: Members (Hard) |
| :--- | :--- | :--- |
| **Complexity** | Low (Single file) | Medium (Separate files) |
| **Isolation** | Shared `.venv` | Dedicated `.venv` |
| **Conflicts** | Fails on version conflicts | Handles version conflicts |
| **Development** | Single context | Fully isolated contexts |

### Recommendation

* Use **Groups** for adding optional features or alternative engines that play well together.
* Use **Members** when moving to a new project "era" with breaking library changes.
