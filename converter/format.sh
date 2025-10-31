#!/bin/bash

set -e

CLANG_FORMAT_FILE="../.clang-format"
TARGET_DIR="./"
IGNORE_DIRS=(
    "build"
    "include/argparse"
)

if [ ! -f "$CLANG_FORMAT_FILE" ]; then
    echo "Error: .clang-format file not found at '$CLANG_FORMAT_FILE'"
    exit 1
fi

echo "Using clang-format config: $CLANG_FORMAT_FILE"
echo "Formatting directory: $TARGET_DIR"
if [ ${#IGNORE_DIRS[@]} -gt 0 ]; then
    echo "Ignoring directories: ${IGNORE_DIRS[*]}"
fi
echo


DIR_TO_FORMAT=$(eval "find \"$TARGET_DIR\" $PRUNE_EXPR -type d -print" | tr '\n' ' ')
for dir in $DIR_TO_FORMAT; do

    skip_dir=false
    for ignore in "${IGNORE_DIRS[@]}"; do
        if [[ "$dir" == *"$ignore"* ]]; then
            skip_dir=true
            break
        fi
    done

    if [ "$skip_dir" == true ]; then
        # echo "Skipping directory: $dir"
        continue
    fi

    echo "Processing directory: $dir"
    files=$(find "$dir" -maxdepth 1 -type f \( -name '*.cpp' -o -name '*.hpp' -o -name '*.hh' \) -print)
    if [ -z "$files" ]; then
        continue
    fi
    for file in $files; do
        echo "  Formatting: $file"
        clang-format -i --style=file:"$CLANG_FORMAT_FILE" "$file"
    done
    
done

echo
echo "✅ Formatting complete."
