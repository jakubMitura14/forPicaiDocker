#!/bin/bash
echo "starttttt a"

set -ex
echo "starttttt b"

script_dir=$(cd $(dirname $0) || exit 1; pwd)

echo "starttttt c"

################################################################################
# Set up headless environment
source $script_dir/start-xorg.sh

echo "starttttt d"

#/usr/bin/x11vnc -forever -rfbport $VNCPORT -display :10 -shared -bg -auth none -nopw
#sleep 1

################################################################################
# start window manager

# Do not start a window manager to make the remote desktop look more like a simple
# remote view rather than a complete remote computer.
#awesome &

################################################################################
# Set jupyter terminal to bash (better auto-complete, etc. than default sh)
echo "starttttt e"

export SHELL=/bin/bash
echo "starttttt f"

################################################################################
# this needs to be last
exec "$@"
echo "starttttt g"
