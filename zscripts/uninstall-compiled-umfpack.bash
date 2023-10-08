#!/bin/bash

read -p "Are you sure [y/n]? " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "sudo rm -rf /usr/local/include/umfpack"
    sudo rm -rf /usr/local/include/umfpack

    echo "sudo rm -rf /usr/local/lib/umfpack"
    sudo rm -rf /usr/local/lib/umfpack

    echo "sudo rm -f /etc/ld.so.conf.d/umfpack.conf"
    sudo rm -f /etc/ld.so.conf.d/umfpack.conf

    echo "sudo ldconfig"
    sudo ldconfig

    echo "DONE"
else
    echo "..cancelled.."
fi
