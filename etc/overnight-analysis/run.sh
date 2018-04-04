#!/bin/bash

cat *.lon | sort | uniq -c  | sort -n -r > count.txt
