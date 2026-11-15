#!/usr/bin/env bash
# Download the ReasAlign defense LoRA from the authors' repository (leolee99/ReasAlign, pinned commit) and verify it.
set -euo pipefail
cd "$(dirname "$0")/.."
COMMIT=e311ffe50d65408d96cc398417fe18f58e3fcfc0
BASE="https://raw.githubusercontent.com/leolee99/ReasAlign/$COMMIT/adapters/defense_model"
DEST=adapters/defense_model
mkdir -p "$DEST"
while read -r sum file; do
  if [ -f "$DEST/$file" ] && echo "$sum  $DEST/$file" | sha256sum -c --status; then continue; fi
  echo "downloading $file"
  curl -fsSL "$BASE/$file" -o "$DEST/$file"
  echo "$sum  $DEST/$file" | sha256sum -c --quiet
done <<'EOF'
c5d8f79dff623452b6c0512d353539197b9061c455d15738df2186f67be48234 adapter_model.safetensors
ccbeeef6ddaa9e3d7a2a2eb6eb546668da2cb28ed85721b265d24469ace0ee34 adapter_config.json
6b9e4e7fb171f92fd137b777cc2714bf87d11576700a1dcd7a399e7bbe39537b tokenizer.json
9f990d1289aa004e3d75a852aa8cf634e1a45c77e5376b2047c7623e027a3161 tokenizer_config.json
994823bd9d0de3b2f59f09f6502a65a228954ca8e5b711534988deb083c37449 special_tokens_map.json
e10ca381b1ccc5cf9db52e371f3b6651576caee0a630b452e2816b2d404d4b65 chat_template.jinja
EOF
echo "adapter ready in $DEST"
