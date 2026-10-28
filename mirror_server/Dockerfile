# Apache 2.4 with Debian's default configuration, as used to serve the mirrors in the
# ProdCrawl experiments: Content-Type from /etc/mime.types by file extension, none
# for unknown extensions, directory listings for folders without index.html, and a
# 301 redirect from /<site> to /<site>/.
FROM debian:trixie-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends apache2 \
 && rm -rf /var/lib/apt/lists/* /var/www/html/index.html
EXPOSE 80
CMD ["apachectl", "-D", "FOREGROUND"]
