#!/usr/bin/env bash

make html
aws s3 sync _build/html/ s3://bdd100k.com/doc --exclude ".DS_Store" --acl public-read
