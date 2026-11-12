{
  description = "Reproducible school data science environment with uv and marimo";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    devDB.url = "github:hermann-p/nix-postgres-dev-db";
    devDB.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, nixpkgs, flake-utils, devDB }:
    flake-utils.lib.eachSystem [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" ] (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
        db = devDB.outputs.packages.${system};
        postgres = pkgs.postgresql_17.withPackages (ps: [
          ps.pgvector
        ]);
        lib = pkgs.lib;
        isLinux = pkgs.stdenv.hostPlatform.isLinux;

        playwrightBrowsers = pkgs.playwright-driver.browsers.override {
          withFirefox = false;
          withWebkit = false;
        };
        playwrightSetup = pkgs.writeShellApplication {
          name = "school-playwright-setup";
          runtimeInputs = [ pkgs.coreutils ];
          text = ''
            # marimo runs `playwright install`; its registry must be writable.
            cache="''${XDG_CACHE_HOME:-$HOME/.cache}/RAG-OERBOTJE/${builtins.baseNameOf (toString playwrightBrowsers)}"
            for browser_dir in "${playwrightBrowsers}"/*; do
              target="$cache/''${browser_dir##*/}"
              mkdir -p "$target"
              for asset in "$browser_dir"/*; do
                ln -sfn "$asset" "$target/''${asset##*/}"
              done
              touch "$target/INSTALLATION_COMPLETE"
            done
            printf '%s\n' "$cache"
          '';
        };
        browserRuntimeHook = ''
          export PLAYWRIGHT_NODEJS_PATH="${lib.getExe pkgs.nodejs}"
          ${lib.optionalString isLinux ''
            PLAYWRIGHT_BROWSERS_PATH="$(${playwrightSetup}/bin/school-playwright-setup)"
            export PLAYWRIGHT_BROWSERS_PATH
            export PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS=true
          ''}
        '';

        cudaRuntimeHook = ''
          export LD_LIBRARY_PATH="${lib.optionalString isLinux "/run/opengl-driver/lib:"}${lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib pkgs.zlib ]}:''${LD_LIBRARY_PATH:-}"
          for lib_dir in "$PWD"/.venv/lib/python*/site-packages/nvidia/*/lib; do
            if [ -d "$lib_dir" ]; then
              export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$lib_dir"
            fi
          done
        '';
        notebookRuntimeHook = browserRuntimeHook + cudaRuntimeHook;

        commonPackages = with pkgs; [
          python312
          nodejs
          zsh
          uv
          process-compose
          direnv
          git
          tmux
          pandoc
          graphviz
          jq
          fzf
          curl
          wget
          htop
          ninja
          cmake
          gcc
          pkg-config
          zlib
        ];

        projectSetup = pkgs.writeShellApplication {
          name = "school-setup";
          runtimeInputs = [ pkgs.uv pkgs.python312 pkgs.coreutils ];
          text = ''
            if [ ! -f pyproject.toml ]; then
              echo "Run this command from the project root." >&2
              exit 1
            fi
            uv sync --locked --python "${pkgs.python312}/bin/python3.12"
          '';
        };

        notebook = pkgs.writeShellApplication {
          name = "notebook";
          runtimeInputs = [ pkgs.uv pkgs.python312 ];
          text = ''
            if [ "$#" -gt 1 ]; then
              echo "Usage: notebook [notebook.py]" >&2
              exit 2
            fi

            if [ ! -f pyproject.toml ]; then
              echo "Run notebook from the project root." >&2
              exit 1
            fi

            export UV_PYTHON_PREFERENCE=only-system
            ${notebookRuntimeHook}
            exec uv run --locked marimo edit "$@" --host 127.0.0.1
          '';
        };

        notebookRun = pkgs.writeShellApplication {
          name = "notebook-run";
          runtimeInputs = [ pkgs.uv pkgs.python312 ];
          text = ''
            if [ "$#" -ne 1 ]; then
              echo "Usage: notebook-run notebook.py" >&2
              exit 2
            fi
            ${notebookRuntimeHook}
            uv run --locked marimo run "$1" --host 127.0.0.1 --no-token
          '';
        };

        fuzzyRun = pkgs.writeShellApplication {
          name = "fuzzy-run";
          runtimeInputs = [ pkgs.fzf pkgs.uv pkgs.python312 pkgs.findutils ];
          text = ''
            ${notebookRuntimeHook}

            if [ ! -f pyproject.toml ] || [ ! -d notebooks ]; then
              echo "Run fuzzy-run from the repository root." >&2
              exit 1
            fi

            selected=$(find notebooks -type f -name '*.py' -print | sort | fzf --prompt='Notebook > ')
            if [ -z "$selected" ]; then
              exit 0
            fi

            exec uv run --locked marimo edit "$selected" --host 127.0.0.1
          '';
        };

        processCompose = pkgs.writeShellApplication {
          name = "school-process-compose";
          runtimeInputs = [ pkgs.process-compose pkgs.uv pkgs.python312 ];
          text = ''
            ${notebookRuntimeHook}
            process-compose --no-server up "$@"
          '';
        };

        notebookConvert = pkgs.writeShellApplication {
          name = "notebook-convert";
          runtimeInputs = [ pkgs.uv pkgs.python312 pkgs.coreutils ];
          text = ''
            if [ "$#" -ne 1 ]; then
              echo "Usage: notebook-convert notebook.ipynb" >&2
              exit 2
            fi

            input=$1
            if [ ! -f "$input" ]; then
              echo "Notebook not found: $input" >&2
              exit 1
            fi

            printf "Name for the new notebook directory (e.g. regression-basics): "
            IFS= read -r name
            if [ -z "$name" ] || [[ ! "$name" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then
              echo "Use letters, numbers, underscores, or hyphens; the name cannot be empty." >&2
              exit 1
            fi

            output_dir="notebooks/$name"
            output="$output_dir/$name.py"
            if [ -e "$output_dir" ]; then
              echo "Refusing to overwrite existing directory: $output_dir" >&2
              exit 1
            fi
            mkdir -p "$output_dir"
            if uv run --locked marimo convert "$input" -o "$output"; then
              printf 'Created %s\n' "$output"
            else
              rm -rf "$output_dir"
              exit 1
            fi
          '';
        };

        dataView = pkgs.writeShellApplication {
          name = "data-view";
          runtimeInputs = [ pkgs.visidata ];
          text = ''
            if [ "$#" -lt 1 ]; then
              echo "Usage: data-view file [file ...]" >&2
              exit 2
            fi
            vd "$@"
          '';
        };

        help = pkgs.writeShellApplication {
          name = "school-help";
          text = ''
            printf '%s\n' \
              'School data science environment' \
              ' ' \
              '  notebook [file.py]       Edit a marimo notebook' \
              '  notebook-run file.py     Run a notebook as a read-only app' \
              '  notebook-convert file    Convert .ipynb into notebooks/<name>/' \
              '  data-view file           Explore tabular data in VisiData' \
              '  drop                     Stage and search downloaded data' \
              '  drop data                Browse staged datasets' \
              '  school-process-compose   Run marimo with the Process Compose TUI' \
              ' ' \
              '  uv add PACKAGE           Add and lock a Python dependency' \
              '  uv sync                  Reproduce the environment from uv.lock' \
              '  nix develop              Enter the default CUDA-enabled shell' \
              '  nix develop .#cpu        Enter the CPU-only runtime shell'
          '';
        };

        commands = [ projectSetup notebook notebookRun fuzzyRun notebookConvert dataView processCompose help ];
        drop = pkgs.writeShellApplication {
          name = "drop";
          runtimeInputs = [ pkgs.python312 pkgs.coreutils ];
          text = ''
            export PYTHONPATH="''${DATASET_INBOX_ROOT:-$PWD}/dataset-inbox/src''${PYTHONPATH:+:$PYTHONPATH}"
            exec python -m dataset_inbox.cli "$@"
          '';
        };
        commandBins = [
          "${drop}/bin"
          "${projectSetup}/bin"
          "${notebook}/bin"
          "${notebookRun}/bin"
          "${fuzzyRun}/bin"
          "${notebookConvert}/bin"
          "${dataView}/bin"
          "${processCompose}/bin"
          "${help}/bin"
        ];
        shellPath = lib.concatStringsSep ":" commandBins;

        baseShell = pkgs.mkShell {
          packages = commonPackages;
          shellHook = ''
            export UV_PYTHON_PREFERENCE=only-system
            export DATASET_INBOX_ROOT="$PWD"
            export UV_PYTHON="${pkgs.python312}/bin/python3.12"
            export UV_PYTHON_DOWNLOADS=never
            ${projectSetup}/bin/school-setup
            export VIRTUAL_ENV="$PWD/.venv"
            ${browserRuntimeHook}
            export LD_LIBRARY_PATH="${lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib pkgs.zlib ]}:''${LD_LIBRARY_PATH:-}"
            export PATH="${lib.makeBinPath [ pkgs.uv ]}:$VIRTUAL_ENV/bin:${shellPath}:$PATH"
            if [ -t 1 ]; then
              printf '\nSchool environment ready. Run school-help for commands.\n\n'
            fi
          '';
        };

        quietShell = pkgs.mkShell {
          packages = commonPackages;
          shellHook = ''
            export UV_PYTHON_PREFERENCE=only-system
            export DATASET_INBOX_ROOT="$PWD"
            ${projectSetup}/bin/school-setup
            export VIRTUAL_ENV="$PWD/.venv"
            ${notebookRuntimeHook}
            export PATH="$VIRTUAL_ENV/bin:${shellPath}:$PATH"
          '';
        };

        cudaShell = pkgs.mkShell {
          packages = commonPackages;
          buildInputs = [
            postgres
            db.start-database
            db.stop-database
            db.psql-wrapped
          ];
          shellHook = ''
            export UV_PYTHON_PREFERENCE=only-system
            export DATASET_INBOX_ROOT="$PWD"
            export UV_PYTHON="${pkgs.python312}/bin/python3.12"
            export UV_PYTHON_DOWNLOADS=never
            export PG_ROOT="$(git rev-parse --show-toplevel)"
            ${projectSetup}/bin/school-setup
            export VIRTUAL_ENV="$PWD/.venv"
            ${notebookRuntimeHook}
            export PATH="${lib.makeBinPath [ pkgs.uv ]}:$VIRTUAL_ENV/bin:${shellPath}:$PATH"
            if [ -t 1 ]; then
              printf '\nSchool CUDA environment ready. Run school-help for commands.\n\n'
            fi
          '';
        };
      in
      {
        packages = {
          inherit notebook notebookRun fuzzyRun notebookConvert dataView processCompose drop;
          default = notebook;
        };

        apps = {
          notebook = flake-utils.lib.mkApp { drv = notebook; };
          notebook-run = flake-utils.lib.mkApp { drv = notebookRun; };
          fuzzy-run = flake-utils.lib.mkApp { drv = fuzzyRun; };
          notebook-convert = flake-utils.lib.mkApp { drv = notebookConvert; };
          data-view = flake-utils.lib.mkApp { drv = dataView; };
          process-compose = flake-utils.lib.mkApp { drv = processCompose; };
        };

        devShells = {
          default = cudaShell;
          cpu = baseShell;
          quiet = quietShell;
          cuda = cudaShell;
        };
        devShell = cudaShell;
      });
}
