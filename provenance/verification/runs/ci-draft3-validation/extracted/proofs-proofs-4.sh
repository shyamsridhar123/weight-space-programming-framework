curl --fail --location --retry 3 \
  https://github.com/leanprover/elan/releases/download/v4.2.4/elan-x86_64-unknown-linux-gnu.tar.gz \
  --output "$RUNNER_TEMP/elan.tar.gz"
printf '%s  %s\n' \
  42b94d4244e8353142c456ec0e4ca6528fd898a6c604d4059f494e706e431f63 \
  "$RUNNER_TEMP/elan.tar.gz" | sha256sum --check
mkdir -p "$RUNNER_TEMP/wsp-elan-installer"
tar -xzf "$RUNNER_TEMP/elan.tar.gz" -C "$RUNNER_TEMP/wsp-elan-installer"
"$RUNNER_TEMP/wsp-elan-installer/elan-init" -y --no-modify-path --default-toolchain none
echo "$HOME/.elan/bin" >> "$GITHUB_PATH"
"$HOME/.elan/bin/elan" toolchain install leanprover/lean4:v4.24.0
