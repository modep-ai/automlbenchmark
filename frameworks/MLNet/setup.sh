#!/usr/bin/env bash
HERE=$(dirname "$0")
MLNET='mlnet'
VERSION=${1:-"latest"}
DOTNET_INSTALL_DIR="$HERE/lib"
MLNET="$DOTNET_INSTALL_DIR/mlnet"
DOTNET="$DOTNET_INSTALL_DIR/dotnet"
SOURCE="https://mlnetcli.blob.core.windows.net/mlnetcli/index.json"

# if version eq latest, set Version to empty string so it will install the latest version.
if [[ "$VERSION" == "latest" ]]; then
    VERSION=""
fi

rm -rf DOTNET_INSTALL_DIR
# install mlnet if necessary
if [[ ! -x "$MLNET" ]]; then
    if [[ ! -x "$DOTNET" ]]; then
        wget -P "$DOTNET_INSTALL_DIR" https://dot.net/v1/dotnet-install.sh
        chmod +x "$DOTNET_INSTALL_DIR/dotnet-install.sh"
        "$DOTNET_INSTALL_DIR/dotnet-install.sh" -c Current --install-dir "$DOTNET_INSTALL_DIR" -Channel 3.1 --verbose
    fi

    $DOTNET --version
    $DOTNET tool install mlnet --add-source "$SOURCE" --version "$VERSION" --tool-path "$DOTNET_INSTALL_DIR"
else
$DOTNET tool update mlnet --add-source "$SOURCE" --version "$VERSION" --tool-path "$DOTNET_INSTALL_DIR"
fi

export DOTNET_ROOT="$DOTNET_INSTALL_DIR"
export MLNET_CLI_HOME="$DOTNET_INSTALL_DIR"

$MLNET --version | grep + | sed -e "s/\(.?*\)+.*/\1/" >> "${HERE}/.installed"

# $ DOTNET_ROOT='./lib' MLNET_CLI_HOME='./lib' lib/mlnet --version
# 16.2.0+511cc1082bef7a4bbb2f83ad88c1c425c932079c
# Use this to get just 16.2.0
# sed 's/+.*//'
