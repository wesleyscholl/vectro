# Homebrew formula for the Vectro CLI.
#
# This file lives at Formula/vectro.rb in the main repo as a canonical source.
# To publish via a Homebrew tap, copy it (or symlink it) into:
#   wesleyscholl/homebrew-tap/Formula/vectro.rb
#
# Users install with:
#   brew tap wesleyscholl/tap
#   brew install vectro
#
# To update the formula after a release:
#   1. Download the release tarball and compute its SHA256:
#        curl -L <tarball_url> | shasum -a 256
#   2. Replace the `url` and `sha256` fields below.
#   3. Bump `version` in the class docstring (cosmetic — Homebrew reads the tag).

class Vectro < Formula
  desc "High-performance embedding compression library — CLI tool"
  homepage "https://github.com/wesleyscholl/vectro"
  # NOTE: Update url + sha256 after each release by fetching the source tarball.
  # The sha256 below is a placeholder — replace before publishing the tap.
  url "https://github.com/wesleyscholl/vectro/archive/refs/tags/v4.8.0.tar.gz"
  sha256 "PLACEHOLDER_UPDATE_AFTER_RELEASE"
  license "MIT"
  head "https://github.com/wesleyscholl/vectro.git", branch: "main"

  depends_on "rust" => :build

  def install
    system "cargo", "install",
           "--locked",
           "--path", "rust/vectro_cli",
           "--root", prefix
  end

  test do
    assert_match "vectro", shell_output("#{bin}/vectro --help")
  end
end
