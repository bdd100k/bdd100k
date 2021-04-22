# BDD100K Documentation

## Build the doc

1. Clone or link bdd100k-doc-media to `media` under this folder.

   ```bash
   git clone git@github.com:bdd100k/doc-media.git
   ln -s doc-media media
   ```

2. Compile the doc

   ```bash
   make html
   ```

## Public doc

The document will be deployed to the public S3 bucket when merged to `master`.

Main website: https://doc.bdd100k.com

S3 static website: https://s3.eu-central-1.amazonaws.com/doc.bdd100k.com/index.html
