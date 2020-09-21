#!/usr/bin/env bash

make html
aws s3 sync _build/html/ s3://doc.bdd100k.com/ --exclude ".DS_Store" --acl public-read
