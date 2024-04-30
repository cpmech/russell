#!/bin/bash

read -p "Are you sure [y/n]? " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "sudo rm -rf /usr/local/include/suitesparse"
    sudo rm -rf /usr/local/include/suitesparse

    echo "sudo rm -rf /usr/local/lib/suitesparse"
    sudo rm -rf /usr/local/lib/suitesparse

    echo "sudo rm -f /etc/ld.so.conf.d/suitesparse.conf"
    sudo rm -f /etc/ld.so.conf.d/suitesparse.conf

    echo "sudo ldconfig"
    sudo ldconfig

    echo "DONE"
else
    echo "..cancelled.."
fi
