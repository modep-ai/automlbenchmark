#!/usr/bin/env bash
HERE=$(dirname "$0")
VERSION=${1:-"stable"}
REPO=${2:-"https://github.com/hyperopt/hyperopt-sklearn.git"}
PKG=${3:-"hyperopt-sklearn"}
if [[ "$VERSION" == "latest" ]]; then
    VERSION="master"
fi

. ${HERE}/../shared/setup.sh ${HERE} true

if [[ "$VERSION" == "stable" ]]; then
    PIP install --no-cache-dir -U ${PKG}
elif [[ "$VERSION" =~ ^[0-9] ]]; then
    # PIP install --no-cache-dir -U ${PKG}==${VERSION}
    PIP install --no-cache-dir -U git+https://github.com/hyperopt/hyperopt-sklearn@${VERSION}
else
#    PIP install --no-cache-dir -e git+${REPO}@${VERSION}#egg=${PKG}
    LIB=$(echo ${PKG} | sed "s/\[.*\]//")
    TARGET_DIR="${HERE}/lib/${LIB}"
    rm -Rf ${TARGET_DIR}
    # git clone --depth 1 --single-branch --branch ${VERSION} --recurse-submodules ${REPO} ${TARGET_DIR}

    # There are no versions of this package, so we let $VERSION be a commit so that we can at least pin.
    # The master branch is cloned, then we checkout the commit named $VERSION.
    git clone --depth 1 --single-branch --branch master --recurse-submodules ${REPO} ${TARGET_DIR}
    cd ${TARGET_DIR}
    git checkout ${VERSION}
    cd ${HERE}

    PIP install -U -e ${HERE}/lib/${PKG}
fi
