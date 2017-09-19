#!/bin/sh

set -e

rm -f ./.coverage.*
tox -c /dist/tox.ini

