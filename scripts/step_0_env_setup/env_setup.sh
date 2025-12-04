#!/bin/bash
sudo sync
# fix core freq
sudo cpupower frequency-set -f 2GHz
# setup IAA
sudo ./configure_iaa_user.sh